<#
.SYNOPSIS
  Launch the Video Processing Portal.

.DESCRIPTION
  Prompts for the portal password and launches the portal via `uv run`.

  On first run, generates and persists a session-signing secret key to
  .portal_secret_key in the repo root (gitignored). Subsequent runs reuse it,
  so existing browser sessions survive restarts.

  Environment overrides honored if already set: PORTAL_HOST, PORTAL_PORT,
  PORTAL_COOKIE_SECURE.
#>

$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$secretKeyPath = Join-Path $repoRoot '.portal_secret_key'

if (-not (Test-Path $secretKeyPath)) {
    Write-Host "First run: generating a portal session secret key at .portal_secret_key"
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    Set-Content -Path $secretKeyPath -Value $key -NoNewline -Encoding ASCII
}

$env:PORTAL_SECRET_KEY = (Get-Content $secretKeyPath -Raw).Trim()

$securePass = Read-Host -Prompt 'Enter portal password' -AsSecureString
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePass)
try {
    $env:PORTAL_PASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
} finally {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if ([string]::IsNullOrEmpty($env:PORTAL_PASSWORD)) {
    Write-Error 'No password entered. Aborting.'
    exit 1
}

$portForBanner = if ($env:PORTAL_PORT) { $env:PORTAL_PORT } else { '8080' }
Write-Host "Launching portal on port $portForBanner (Ctrl+C to stop) ..."

uv run video-processing-portal
