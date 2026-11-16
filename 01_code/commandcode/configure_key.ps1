$ErrorActionPreference = 'Stop'
$secretDir = Join-Path $env:LOCALAPPDATA 'NIDSResearchSecrets'
New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
$secret = Read-Host 'Paste new CommandCode key (hidden)' -AsSecureString
$secret | ConvertFrom-SecureString | Set-Content -LiteralPath (Join-Path $secretDir 'commandcode.dpapi') -Encoding ASCII
Write-Host 'Saved encrypted for this Windows user. No key is stored in the project.'
