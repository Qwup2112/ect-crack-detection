# Creates "ECT Crack Chat.lnk" (opens the desktop program, with its icon).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File app\assets\make_shortcut.ps1 [-Dest <folder>]
# Default destination: the project folder. Pass -Dest "$env:USERPROFILE\Desktop" for the desktop.
param([string]$Dest)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Dest) { $Dest = $root }
$lnk = Join-Path $Dest "ECT Crack Chat.lnk"
$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($lnk)
$s.TargetPath = Join-Path $root ".venv\Scripts\pythonw.exe"
$s.Arguments = "-m app.desktop"
$s.WorkingDirectory = $root
$s.IconLocation = (Join-Path $root "app\assets\ect.ico") + ",0"
$s.Description = "ECT Crack Chat - tro ly ve va nhan dien vet nut ECT"
$s.Save()
$lnk
