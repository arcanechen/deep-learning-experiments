param(
    [ValidateSet('train','evaluate','test')][string]$Action = 'train',
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = 'Stop'
$pythonPath = 'D:\Anaconda\envs\ai-lab\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $PSScriptRoot
try {
    if ($Action -eq 'test') {
        & $pythonPath -m unittest discover -s tests -v
    } elseif ($Action -eq 'evaluate') {
        & $pythonPath -u evaluate.py @ExtraArgs
    } else {
        & $pythonPath -u train.py @ExtraArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally { Pop-Location }
