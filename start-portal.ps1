<#
.SYNOPSIS
  Launch the Video Processing Portal with a public localhost.run tunnel.

.DESCRIPTION
  One-command portal startup:
    1. Generates and persists a session-signing secret key on first run
       (.portal_secret_key, gitignored). Reuses it thereafter so existing
       browser sessions survive restarts.
    2. Prompts for the portal password.
    3. Starts an SSH reverse tunnel to localhost.run in the background and
       prints the public https URL to share with collaborators.
    4. Launches the portal in the foreground.
    5. Stops the tunnel when the portal exits.

  Environment overrides honored if already set: PORTAL_HOST, PORTAL_PORT,
  PORTAL_COOKIE_SECURE.

.PARAMETER NoTunnel
  Skip the public tunnel. Use this for local-only testing on http://localhost:8080.
#>

param(
    [switch]$NoTunnel
)

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

$tunnelProc = $null
if (-not $NoTunnel) {
    if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
        Write-Warning "ssh.exe not found on PATH. Skipping public tunnel; portal will be local-only."
        Write-Warning "Install the Windows OpenSSH client (Settings -> Apps -> Optional features) to enable tunnels."
    } else {
        $tunnelLog = Join-Path $repoRoot '.portal_tunnel.log'
        $tunnelErrLog = Join-Path $repoRoot '.portal_tunnel.err.log'
        Remove-Item -ErrorAction SilentlyContinue $tunnelLog, $tunnelErrLog

        Write-Host "Starting public tunnel via localhost.run ..."
        $tunnelProc = Start-Process -FilePath 'ssh' -ArgumentList @(
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'ServerAliveInterval=60',
            '-o', 'ExitOnForwardFailure=yes',
            '-R', "80:127.0.0.1:$portForBanner",
            'nokey@localhost.run'
        ) -RedirectStandardOutput $tunnelLog `
          -RedirectStandardError $tunnelErrLog `
          -NoNewWindow -PassThru

        $tunnelUrl = $null
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            if ($tunnelProc.HasExited) {
                Write-Warning "Tunnel process exited unexpectedly. See $tunnelErrLog"
                $tunnelProc = $null
                break
            }
            if (Test-Path $tunnelLog) {
                $content = Get-Content -Raw $tunnelLog -ErrorAction SilentlyContinue
                if ($content -and ($content -match 'https://[a-z0-9-]+\.lhr\.life')) {
                    $tunnelUrl = $matches[0]
                    break
                }
            }
        }

        if ($tunnelUrl) {
            Write-Host ""
            Write-Host "============================================================"
            Write-Host " Public URL: $tunnelUrl"
            Write-Host " Share this URL and the portal password with collaborators."
            Write-Host "============================================================"
            Write-Host ""
        } elseif ($tunnelProc) {
            Write-Warning "Tunnel started but no URL detected within 30s. Check $tunnelLog"
        }
    }
}

Write-Host "Launching portal on port $portForBanner (Ctrl+C to stop) ..."

try {
    uv run video-processing-portal
} finally {
    if ($tunnelProc -and -not $tunnelProc.HasExited) {
        Write-Host "Stopping tunnel ..."
        Stop-Process -Id $tunnelProc.Id -Force -ErrorAction SilentlyContinue
    }
}
