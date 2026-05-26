<#
.SYNOPSIS
  Launch the Video Processing Portal with a public tunnel.

.DESCRIPTION
  One-command portal startup:
    1. Generates and persists a session-signing secret key on first run
       (.portal_secret_key, gitignored). Reuses it thereafter so existing
       browser sessions survive restarts.
    2. Prompts for the portal password.
    3. Starts a public tunnel in the background and prints the URL:
         - Prefers `cloudflared` (Cloudflare Quick Tunnel, *.trycloudflare.com).
           The URL is stable for the lifetime of the cloudflared process, so
           it does not rotate mid-session.
         - Falls back to `ssh` to `localhost.run` (*.lhr.life) if cloudflared
           is not installed. Anonymous lhr tunnels rotate ~hourly; the user
           will need to refresh the URL if they stay connected past that.
         - Falls back to local-only if neither tool is available.
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

function Start-CloudflareTunnel {
    param([string]$Port, [string]$LogPath, [string]$ErrLogPath)

    Write-Host "Starting public tunnel via Cloudflare (trycloudflare.com) ..."
    $proc = Start-Process -FilePath 'cloudflared' -ArgumentList @(
        'tunnel', '--no-autoupdate', '--url', "http://localhost:$Port"
    ) -RedirectStandardOutput $LogPath `
      -RedirectStandardError $ErrLogPath `
      -NoNewWindow -PassThru

    $url = $null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) {
            Write-Warning "cloudflared exited unexpectedly. See $ErrLogPath"
            return @{ Process = $null; Url = $null }
        }
        # cloudflared writes the URL to stderr in a banner; check both streams.
        $content = ''
        if (Test-Path $LogPath) { $content += (Get-Content -Raw $LogPath -ErrorAction SilentlyContinue) }
        if (Test-Path $ErrLogPath) { $content += (Get-Content -Raw $ErrLogPath -ErrorAction SilentlyContinue) }
        if ($content -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
            $url = $matches[0]
            break
        }
    }
    return @{ Process = $proc; Url = $url }
}

function Start-LocalhostRunTunnel {
    param([string]$Port, [string]$LogPath, [string]$ErrLogPath)

    Write-Host "Starting public tunnel via localhost.run ..."
    $proc = Start-Process -FilePath 'ssh' -ArgumentList @(
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'ServerAliveInterval=60',
        '-o', 'ExitOnForwardFailure=yes',
        '-R', "80:127.0.0.1:$Port",
        'nokey@localhost.run'
    ) -RedirectStandardOutput $LogPath `
      -RedirectStandardError $ErrLogPath `
      -NoNewWindow -PassThru

    $url = $null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) {
            Write-Warning "ssh tunnel exited unexpectedly. See $ErrLogPath"
            return @{ Process = $null; Url = $null }
        }
        if (Test-Path $LogPath) {
            $content = Get-Content -Raw $LogPath -ErrorAction SilentlyContinue
            if ($content -and ($content -match 'https://[a-z0-9-]+\.lhr\.life')) {
                $url = $matches[0]
                break
            }
        }
    }
    return @{ Process = $proc; Url = $url }
}

$tunnelProc = $null
if (-not $NoTunnel) {
    $tunnelLog = Join-Path $repoRoot '.portal_tunnel.log'
    $tunnelErrLog = Join-Path $repoRoot '.portal_tunnel.err.log'
    Remove-Item -ErrorAction SilentlyContinue $tunnelLog, $tunnelErrLog

    $result = $null
    if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
        $result = Start-CloudflareTunnel -Port $portForBanner -LogPath $tunnelLog -ErrLogPath $tunnelErrLog
    } elseif (Get-Command ssh -ErrorAction SilentlyContinue) {
        Write-Host "cloudflared not found; falling back to ssh + localhost.run (URL rotates ~hourly)."
        $result = Start-LocalhostRunTunnel -Port $portForBanner -LogPath $tunnelLog -ErrLogPath $tunnelErrLog
    } else {
        Write-Warning "Neither cloudflared nor ssh found on PATH. Portal will be local-only."
        Write-Warning "Install cloudflared (https://github.com/cloudflare/cloudflared/releases) for a stable tunnel."
    }

    if ($result) {
        $tunnelProc = $result.Process
        if ($result.Url) {
            Write-Host ""
            Write-Host "============================================================"
            Write-Host " Public URL: $($result.Url)"
            Write-Host " Share this URL and the portal password with collaborators."
            Write-Host "============================================================"
            Write-Host ""
        } elseif ($tunnelProc) {
            Write-Warning "Tunnel started but no URL detected within 30s. Check $tunnelLog and $tunnelErrLog"
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
