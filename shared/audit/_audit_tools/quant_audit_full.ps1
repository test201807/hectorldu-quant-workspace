param(
  [string]$ROOT = "C:\Quant",
  [string]$OUTDIR = "",
  [string]$MODEL_FILES = "gpt-5.2-mini",
  [string]$MODEL_PLAN  = "gpt-5.2",
  [ValidateSet("low","medium","high")] [string]$EFFORT = "low",

  [int]$MAX_DEPTH = 8,
  [int]$MAX_FILES = 50000,

  # Enviamos contenido SOLO de archivos texto razonables (code/config/docs/notebooks).
  # TODO lo demás (parquet/csv/binarios) va como metadata (nombre/tamaño/hash), no como contenido.
  [int]$MAX_TEXT_KB = 1024,
  [int]$MAX_CHARS_PER_FILE = 20000,
  [int]$MAX_NOTEBOOK_CELLS = 120,

  # Tamaño de batch de prompt hacia la API
  [int]$BATCH_MAX_CHARS = 120000,

  [switch]$SKIP_FILE_CLASSIFY,
  [switch]$SKIP_PLAN
)

$ErrorActionPreference = "Stop"

if (-not $OUTDIR -or $OUTDIR.Trim() -eq "") {
  $OUTDIR = Join-Path $ROOT "_audit_out"
}

if (-not (Test-Path $ROOT)) { throw "ROOT no existe: $ROOT" }
if (-not $env:OPENAI_API_KEY) { throw "Falta OPENAI_API_KEY en variables de entorno" }

New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

$manifestPath = Join-Path $OUTDIR "manifest.jsonl"
$digestsPath  = Join-Path $OUTDIR "digests.jsonl"
$errorsPath   = Join-Path $OUTDIR "errors.log"

$filesMapPath = Join-Path $OUTDIR "openai_files_map.json"
$planPath     = Join-Path $OUTDIR "openai_plan.json"
$movePlanPath = Join-Path $OUTDIR "move_plan.ps1"

Remove-Item -Force -ErrorAction SilentlyContinue $manifestPath, $digestsPath, $errorsPath, $filesMapPath, $planPath, $movePlanPath

function Safe-LogError([string]$msg) {
  $ts = (Get-Date).ToString("s")
  "$ts`t$msg" | Add-Content -Encoding UTF8 $errorsPath
}

function Get-Depth([string]$base, [string]$full) {
  $rel = $full.Substring($base.Length).TrimStart('\')
  if ($rel -eq "") { return 0 }
  return ($rel.Split('\').Count)
}

$excludeDirNames = @(
  ".git",".venv","venv","__pycache__","node_modules",
  ".mypy_cache",".pytest_cache",".ruff_cache",".idea",".vscode",
  "site-packages","dist","build",".tox",".cache"
)

function Is-ExcludedPath([string]$path) {
  foreach ($d in $excludeDirNames) {
    if ($path -match [regex]::Escape("\$d\")) { return $true }
    if ($path.EndsWith("\$d", [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
  }
  return $false
}

function Kind-FromExt([string]$ext) {
  if ($null -eq $ext) { $ext = "" }
  $ext = $ext.ToLowerInvariant()

  if ($ext -match '\.(py|ps1|psm1|sql|js|ts|tsx|jsx|mql5|mq5|mqh|r|ipynb)$') { return "code" }
  if ($ext -match '\.(json|toml|yml|yaml|ini|cfg|conf)$') { return "config" }
  if ($ext -match '\.(md|txt|rst)$') { return "docs" }
  if ($ext -match '\.(csv|parquet|arrow|feather|pkl|pickle|h5|hdf5|db)$') { return "data" }
  if ($ext -match '\.(zip|7z|rar|gz|bz2|tar)$') { return "archive" }
  if ($ext -match '\.(dll|exe|pyd|so|bin)$') { return "binary" }
  return "other"
}

function Sanitize([string]$s) {
  if ($null -eq $s) { return "" }
  $s = [regex]::Replace($s, "[\u0000-\u0008\u000B\u000C\u000E-\u001F]", "")
  $s = $s -replace "\uFFFD", ""
  return $s
}

function Read-Utf8Text([string]$path, [int]$maxChars) {
  $bytes = [System.IO.File]::ReadAllBytes($path)
  $text = [System.Text.Encoding]::UTF8.GetString($bytes)   # reemplaza inválidos sin reventar
  $text = Sanitize $text
  if ($text.Length -gt $maxChars) { $text = $text.Substring(0, $maxChars) }
  return $text
}

function Extract-IpynbText([string]$nbRaw, [int]$maxCells, [int]$maxChars) {
  try {
    $nb = $nbRaw | ConvertFrom-Json -Depth 200
    if (-not $nb.cells) {
      if ($nbRaw.Length -gt $maxChars) { return $nbRaw.Substring(0, $maxChars) }
      return $nbRaw
    }

    $sb = New-Object System.Text.StringBuilder
    $i = 0
    foreach ($c in $nb.cells) {
      if ($i -ge $maxCells) { break }
      $ctype = $c.cell_type
      $src = $c.source
      if ($src -is [System.Array]) { $src = ($src -join "") }
      if (-not $src) { $i++; continue }

      [void]$sb.AppendLine("### cell[$i] ($ctype)")
      [void]$sb.AppendLine($src)
      [void]$sb.AppendLine("")
      $i++

      if ($sb.Length -ge $maxChars) { break }
    }

    $out = $sb.ToString()
    if ($out.Length -gt $maxChars) { $out = $out.Substring(0, $maxChars) }
    return $out
  } catch {
    if ($nbRaw.Length -gt $maxChars) { return $nbRaw.Substring(0, $maxChars) }
    return $nbRaw
  }
}

function Extract-PyOutline([string]$text) {
  $imports = @()
  $defs = @()

  foreach ($m in [regex]::Matches($text, "(?m)^\s*(from\s+\S+\s+import\s+.+|import\s+.+)\s*$")) {
    $imports += $m.Value.Trim()
    if ($imports.Count -ge 80) { break }
  }

  foreach ($m in [regex]::Matches($text, "(?m)^\s*(def|class)\s+([A-Za-z_]\w*)\s*\(")) {
    $defs += ("{0}:{1}" -f $m.Groups[1].Value, $m.Groups[2].Value)
    if ($defs.Count -ge 120) { break }
  }

  return @{ imports = $imports; defs = $defs }
}

function Extract-MdHeadings([string]$text) {
  $heads = @()
  foreach ($m in [regex]::Matches($text, "(?m)^\s*#{1,6}\s+.+$")) {
    $heads += $m.Value.Trim()
    if ($heads.Count -ge 80) { break }
  }
  return $heads
}

function Get-Sha256([string]$path, [int64]$maxHashBytes = 104857600) {
  # hash solo si <= 100MB para no matar tiempo
  try {
    $len = (Get-Item -LiteralPath $path).Length
    if ($len -gt $maxHashBytes) { return $null }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
  } catch { return $null }
}

function Extract-OutputText($resp) {
  try {
    foreach ($o in $resp.output) {
      if ($o.type -eq "message") {
        foreach ($c in $o.content) {
          if ($c.type -eq "output_text") { return $c.text }
        }
      }
    }
  } catch {}
  return $null
}

Write-Host "ROOT    : $ROOT"
Write-Host "OUTDIR  : $OUTDIR"
Write-Host "MODELS  : files=$MODEL_FILES | plan=$MODEL_PLAN"
Write-Host ""

# =========================
# 1) Crawl + manifest + digests
# =========================
Write-Host "📦 Indexando archivos..."
$all = Get-ChildItem -Path $ROOT -File -Recurse -Force -ErrorAction SilentlyContinue |
  Select-Object FullName, Length, LastWriteTime, Extension

if ($all.Count -gt $MAX_FILES) {
  $all = $all | Sort-Object LastWriteTime -Descending | Select-Object -First $MAX_FILES
}

$files = New-Object System.Collections.Generic.List[object]
foreach ($f in $all) {
  try {
    if (Is-ExcludedPath $f.FullName) { continue }
    if ((Get-Depth $ROOT $f.FullName) -gt $MAX_DEPTH) { continue }
    $files.Add($f) | Out-Null
  } catch {}
}

Write-Host ("Archivos seleccionados: {0}" -f $files.Count)

foreach ($f in $files) {
  $full = $f.FullName
  $rel  = $full.Substring($ROOT.Length).TrimStart('\')
  $ext  = ""
  if ($null -ne $f.Extension) { $ext = $f.Extension.ToLowerInvariant() }
  $kind = Kind-FromExt $ext
  $size = [int64]$f.Length
  $sha  = Get-Sha256 $full

  $mrec = [pscustomobject]@{
    path = $full
    rel  = $rel
    ext  = $(if ($ext -and $ext.Trim() -ne "") { $ext } else { "(noext)" })
    kind = $kind
    size_bytes = $size
    size_kb = [math]::Round($size / 1KB, 1)
    last_write = $f.LastWriteTime.ToString("s")
    sha256 = $sha
  }
  ($mrec | ConvertTo-Json -Depth 6 -Compress) | Add-Content -Encoding UTF8 $manifestPath

  # digests: solo texto razonable (pero metadata de TODO ya está en manifest)
  $isTextCandidate = $kind -in @("code","config","docs")
  if (-not $isTextCandidate) { continue }
  if ($size -gt ($MAX_TEXT_KB * 1KB)) { continue }

  try {
    $text = Read-Utf8Text $full $MAX_CHARS_PER_FILE
    if ($ext -eq ".ipynb") {
      $text = Extract-IpynbText $text $MAX_NOTEBOOK_CELLS $MAX_CHARS_PER_FILE
    }

    $outline = $null
    $headings = $null
    if ($ext -eq ".py") { $outline = Extract-PyOutline $text }
    elseif ($ext -eq ".md") { $headings = Extract-MdHeadings $text }

    $drec = [pscustomobject]@{
      path = $full
      rel  = $rel
      kind = $kind
      ext  = $(if ($ext -and $ext.Trim() -ne "") { $ext } else { "(noext)" })
      size_kb = [math]::Round($size / 1KB, 1)
      last_write = $f.LastWriteTime.ToString("s")
      headings = $headings
      outline  = $outline
      preview  = $text
    }
    ($drec | ConvertTo-Json -Depth 10 -Compress) | Add-Content -Encoding UTF8 $digestsPath
  } catch {
    Safe-LogError ("Digest fail: {0} :: {1}" -f $full, $_.Exception.Message)
  }
}

Write-Host "✅ manifest: $manifestPath"
Write-Host "✅ digests : $digestsPath"
Write-Host "✅ errors  : $errorsPath"
Write-Host ""

# =========================
# 2) API: clasificar archivos (usa digests)
# =========================
if (-not $SKIP_FILE_CLASSIFY) {

  if (-not (Test-Path $digestsPath)) { throw "No existe digests.jsonl: $digestsPath" }
  $lines = Get-Content -Encoding UTF8 $digestsPath

  Write-Host ("📤 Enviando a API (digests): {0} archivos..." -f $lines.Count)

  $batches = New-Object System.Collections.Generic.List[object]
  $cur = New-Object System.Collections.Generic.List[string]
  $curChars = 0

  foreach ($ln in $lines) {
    $ln2 = Sanitize $ln
    $len = $ln2.Length
    if (($curChars + $len + 1) -gt $BATCH_MAX_CHARS -and $cur.Count -gt 0) {
      $batches.Add(@($cur)) | Out-Null
      $cur = New-Object System.Collections.Generic.List[string]
      $curChars = 0
    }
    $cur.Add($ln2) | Out-Null
    $curChars += ($len + 1)
  }
  if ($cur.Count -gt 0) { $batches.Add(@($cur)) | Out-Null }

  Write-Host ("Batches: {0}" -f $batches.Count)

  $headers = @{
    "Authorization" = "Bearer $($env:OPENAI_API_KEY)"
    "Content-Type"  = "application/json; charset=utf-8"
  }
  $endpoint = "https://api.openai.com/v1/responses"

  $instructions = @"
Eres un auditor técnico para repos cuant/trading en Windows.

INPUT: Recibirás N líneas JSONL; cada línea es un "digest" de un archivo:
{ path, rel, kind, ext, size_kb, last_write, headings?, outline?, preview }

TAREA:
- Para CADA digest, devuelve:
  - category: data_engine | filters | research | strategies | utils | docs | config | env | legacy | data | other
  - summary: 1-2 líneas (qué hace/contiene)
  - confidence: 0..1
  - move_hint: carpeta superior sugerida (ej: "10_data_engine", "20_filters", "30_research", "40_strategies", "90_legacy", "docs", "tools", "config")
  - red_flags: lista breve (paths hardcodeados, outputs fuera de contrato, duplicados, etc.)

REGLA CRÍTICA:
- No inventes contenido. Si el preview es corto o no concluye, baja confidence y dilo.
- No sugieras mover datasets grandes (parquet/csv); eso va a "data/" por reglas, no por preview.

OUTPUT: SOLO JSON válido con esta forma exacta:
{
  "files": [
    {
      "path": "...",
      "rel": "...",
      "category": "...",
      "confidence": 0.0,
      "summary": "...",
      "move_hint": "...",
      "red_flags": ["..."]
    }
  ]
}
"@

  $allFiles = New-Object System.Collections.Generic.List[object]

  for ($i=0; $i -lt $batches.Count; $i++) {
    $payloadText = (($batches[$i]) -join "`n")

    $bodyObj = @{
      model = $MODEL_FILES
      store = $false
      reasoning = @{ effort = $EFFORT }
      input = @(
        @{
          role="system"
          content=@(@{ type="input_text"; text=$instructions })
        },
        @{
          role="user"
          content=@(@{ type="input_text"; text=$payloadText })
        }
      )
    }

    $bodyJson  = Sanitize ($bodyObj | ConvertTo-Json -Depth 12)
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($bodyJson)

    Write-Host ("📡 Batch {0}/{1}..." -f ($i+1), $batches.Count)

    $attempt = 0
    $done = $false
    while (-not $done -and $attempt -lt 4) {
      $attempt++
      try {
        $resp = Invoke-RestMethod -Method Post -Uri $endpoint -Headers $headers -Body $bodyBytes
        $outText = Extract-OutputText $resp
        if (-not $outText) { throw "No output_text en respuesta." }

        $obj = $outText | ConvertFrom-Json
        foreach ($f in $obj.files) { $allFiles.Add($f) | Out-Null }
        $done = $true
      } catch {
        Safe-LogError ("OPENAI batch fail {0} attempt {1} :: {2}" -f $i, $attempt, $_.Exception.Message)
        Start-Sleep -Milliseconds (400 * $attempt)
      }
    }

    Start-Sleep -Milliseconds 150
  }

  @{ files = $allFiles } | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $filesMapPath
  Write-Host "✅ openai_files_map.json: $filesMapPath"
  Write-Host ""
}

# =========================
# 3) API: plan global + estructura + move_plan
# =========================
if (-not $SKIP_PLAN) {

  Write-Host "🧠 Generando plan global con la API..."

  $manifestSample = (Get-Content -Encoding UTF8 $manifestPath | Select-Object -First 400) -join "`n"
  $filesMapText = ""
  if (Test-Path $filesMapPath) {
    $filesMapText = Get-Content -Raw -Encoding UTF8 $filesMapPath
    if ($filesMapText.Length -gt 120000) { $filesMapText = $filesMapText.Substring(0,120000) }
  }

  # intenta incluir contratos si existen dentro de ROOT
  $contractText = ""
  try {
    $contractFiles = Get-ChildItem -Path $ROOT -Recurse -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match 'PATH_CONTRACT' -and $_.Extension -in @(".md",".txt") } |
      Select-Object -First 5
    foreach ($cf in $contractFiles) {
      $t = Read-Utf8Text $cf.FullName 40000
      $contractText += ("`n--- CONTRACT FILE: {0} ---`n{1}`n" -f $cf.FullName, $t)
    }
    $contractText = Sanitize $contractText
    if ($contractText.Length -gt 120000) { $contractText = $contractText.Substring(0,120000) }
  } catch {}

  $planInstructions = @"
Eres un arquitecto/a de repos cuant/trading en Windows.

OBJETIVO:
- Proponer una estructura limpia de carpetas para ROOT, minimizando riesgo de romper contratos.
- Entregar plan de reorganización en 3 fases:
  (1) Ordenar sin romper nada (solo agrupar / crear carpetas y junctions si hace falta)
  (2) Cerrar contratos (paths consistentes, un data_root, outputs estandar)
  (3) Preparar StrategyLab (estrategias por activo) sobre assets filtrados

INPUTS:
(1) MANIFEST sample (metadata de archivos)
(2) FILES MAP (clasificación por archivo si existe)
(3) CONTRACTS si se encontraron (respetar)

OUTPUT: SOLO JSON válido con forma exacta:
{
  "proposed_structure": {
    "root_folders": ["..."],
    "rules": ["..."]
  },
  "projects": [
    { "path_hint": "...", "type": "...", "confidence": 0.0, "notes": ["..."] }
  ],
  "risks": [
    { "risk": "...", "severity": "low|med|high", "why": "...", "mitigation": "..." }
  ],
  "phases": {
    "phase1": ["..."],
    "phase2": ["..."],
    "phase3": ["..."]
  },
  "moves": [
    {
      "from_rel": "...",
      "to_rel": "...",
      "method": "move|junction|keep",
      "reason": "..."
    }
  ]
}
"@

  $headers = @{
    "Authorization" = "Bearer $($env:OPENAI_API_KEY)"
    "Content-Type"  = "application/json; charset=utf-8"
  }
  $endpoint = "https://api.openai.com/v1/responses"

  $planPayload = @"
### CONTRACTS (si existen)
$contractText

### MANIFEST (sample)
$manifestSample

### FILES_MAP (si existe)
$filesMapText
"@

  $bodyObj = @{
    model = $MODEL_PLAN
    store = $false
    reasoning = @{ effort = $EFFORT }
    input = @(
      @{
        role="system"
        content=@(@{ type="input_text"; text=$planInstructions })
      },
      @{
        role="user"
        content=@(@{ type="input_text"; text=$planPayload })
      }
    )
  }

  $bodyJson  = Sanitize ($bodyObj | ConvertTo-Json -Depth 12)
  $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($bodyJson)

  $resp = Invoke-RestMethod -Method Post -Uri $endpoint -Headers $headers -Body $bodyBytes
  $outText = Extract-OutputText $resp
  if (-not $outText) { throw "No output_text en respuesta del plan." }

  $outText | Set-Content -Encoding UTF8 $planPath
  Write-Host "✅ openai_plan.json: $planPath"

  # Genera move_plan.ps1 desde moves[]
  try {
    $plan = $outText | ConvertFrom-Json

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("`$ErrorActionPreference = 'Stop'")
    [void]$sb.AppendLine("`$ROOT = '$ROOT'")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("# ========================================================")
    [void]$sb.AppendLine("# MOVE PLAN (GENERADO) — revísalo antes de ejecutar")
    [void]$sb.AppendLine("# method: keep | move | junction")
    [void]$sb.AppendLine("# ========================================================")
    [void]$sb.AppendLine("")

    foreach ($m in $plan.moves) {
      $from = $m.from_rel
      $to   = $m.to_rel
      $method = $m.method

      if (-not $from -or -not $to -or -not $method) { continue }

      $fromAbs = "Join-Path `$ROOT '$from'"
      $toAbs   = "Join-Path `$ROOT '$to'"

      if ($method -eq "keep") {
        [void]$sb.AppendLine("# KEEP: $from  -> $to  ($($m.reason))")
        continue
      }

      if ($method -eq "move") {
        [void]$sb.AppendLine("# MOVE: $from  -> $to  ($($m.reason))")
        [void]$sb.AppendLine("`$src = $fromAbs")
        [void]$sb.AppendLine("`$dst = $toAbs")
        [void]$sb.AppendLine("New-Item -ItemType Directory -Force -Path (Split-Path -Parent `$dst) | Out-Null")
        [void]$sb.AppendLine("if (Test-Path -LiteralPath `$src) { Move-Item -LiteralPath `$src -Destination `$dst -Force }")
        [void]$sb.AppendLine("")
        continue
      }

      if ($method -eq "junction") {
        [void]$sb.AppendLine("# JUNCTION: $from  -> $to  ($($m.reason))")
        [void]$sb.AppendLine("`$src = $fromAbs")
        [void]$sb.AppendLine("`$dst = $toAbs")
        [void]$sb.AppendLine("New-Item -ItemType Directory -Force -Path `$dst | Out-Null")
        [void]$sb.AppendLine("if (Test-Path -LiteralPath `$src) {")
        [void]$sb.AppendLine("  # si existe, lo movemos a dst y dejamos junction en src")
        [void]$sb.AppendLine("  if (-not (Test-Path -LiteralPath `$dst)) { New-Item -ItemType Directory -Force -Path `$dst | Out-Null }")
        [void]$sb.AppendLine("  robocopy `$src `$dst /E /MOVE | Out-Null")
        [void]$sb.AppendLine("  if (Test-Path -LiteralPath `$src) { Remove-Item -LiteralPath `$src -Force -Recurse -ErrorAction SilentlyContinue }")
        [void]$sb.AppendLine("  cmd /c `"mklink /J `"$(`$src)`" `"$(`$dst)`"`" | Out-Null")
        [void]$sb.AppendLine("}")
        [void]$sb.AppendLine("")
        continue
      }
    }

    $sb.ToString() | Set-Content -Encoding UTF8 $movePlanPath
    Write-Host "✅ move_plan.ps1: $movePlanPath"
  } catch {
    Safe-LogError ("move_plan generation failed :: " + $_.Exception.Message)
  }
}

Write-Host ""
Write-Host "✅ LISTO. Revisa en:"
Write-Host " - $OUTDIR"
Write-Host "   - manifest.jsonl"
Write-Host "   - digests.jsonl"
Write-Host "   - openai_files_map.json"
Write-Host "   - openai_plan.json"
Write-Host "   - move_plan.ps1"
