param(
    [string]$CaptureRoot = (Join-Path $HOME 'nordicai\drone-captures'),
    [string]$Data = '',
    [string]$Python = 'python',
    [string]$RunName = 'validation_finetune_v1',
    [string]$Device = '0',
    [int]$Epochs = 100,
    [int]$Batch = 4,
    [switch]$Resume
)
# User-invoked only. Foreground execution with persistent logs, no prompts/retries.
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must be a simple directory name.' }
if ($Epochs -lt 1 -or $Batch -lt 1) { throw 'Epochs and Batch must be positive.' }
if (-not $Data) { $Data = Join-Path $CaptureRoot 'combined_finetune\data.yaml' }
$Data = (Resolve-Path -LiteralPath $Data).Path
$runDirectory = Join-Path $PSScriptRoot "runs\detect\$RunName"
$baseline = Join-Path $PSScriptRoot 'runs\detect\helsinki_yolov8n_multires_rot\weights\best.pt'
$checkpoint = if ($Resume) { Join-Path $runDirectory 'weights\last.pt' } else { $baseline }
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing checkpoint: $checkpoint" }
if (-not $Resume -and (Test-Path -LiteralPath $runDirectory)) { throw 'Run directory exists; use -Resume or a new -RunName.' }

Push-Location $PSScriptRoot
try {
    & $Python teammate_preflight.py --capture-root $CaptureRoot --weights $checkpoint --dataset-yaml $Data --require-dataset
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed.' }
    $logDirectory = Join-Path $PSScriptRoot 'handoff_logs'
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $log = Join-Path $logDirectory ("{0}_{1}.log" -f $RunName, (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    # Windows PowerShell can treat native stderr progress as errors; preserve it
    # in the log and use the native exit code to decide success.
    $ErrorActionPreference = 'Continue'
    if ($Resume) {
        & $Python -u resume_validation_finetune.py --checkpoint $checkpoint --data $Data --run-dir $runDirectory --device $Device --batch $Batch 2>&1 | Tee-Object -FilePath $log
    } else {
        & $Python -u train_yolo.py --model $baseline --data $Data --name $RunName --device $Device --epochs $Epochs --batch $Batch 2>&1 | Tee-Object -FilePath $log
    }
    $trainingExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($trainingExit -ne 0) { throw "Training exited with code $trainingExit. See $log" }
    Write-Host "Completed. Run: $runDirectory ; log: $log"
} finally {
    Pop-Location
}
