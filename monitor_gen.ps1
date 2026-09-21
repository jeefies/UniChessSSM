$remoteUser = "jeefy"
$remoteHost = "172.16.2.12"
$genDir = "/home/jeefy/UniChess/SSM/runs/stage_b_gen_round2"
$checkCmd = "test -f $genDir/manifest.json && cat $genDir/manifest.json || echo 'NOT_DONE'"
$pollSeconds = 120

Write-Host "Round2 generation monitor started. Polling every ${pollSeconds}s..."
Write-Host "Expected completion: ~22:30-23:00"

while ($true) {
    $result = ssh $remoteUser@$remoteHost $checkCmd 2>$null
    if ($result -and $result -ne "NOT_DONE" -and $result -match 'games') {
        try {
            $manifest = $result | ConvertFrom-Json
            $gen = $manifest.gen
            $gamesDone = $gen.games
            $elapsed = [math]::Round($gen.elapsed_s / 60, 1)
            $gamesPerS = [math]::Round($gen.games_per_s, 3)
            
            $title = "UniChess Round2 Generation Complete!"
            $msg = "$gamesDone games generated in ${elapsed}min (${gamesPerS} games/s)"
            
            # PowerShell balloon notification
            Add-Type -AssemblyName System.Windows.Forms
            $notify = New-Object System.Windows.Forms.NotifyIcon
            $notify.Icon = [System.Drawing.SystemIcons]::Information
            $notify.BalloonTipTitle = $title
            $notify.BalloonTipText = $msg
            $notify.Visible = $true
            $notify.ShowBalloonTip(10000)
            
            # Also write to console with beep
            Write-Host "`a`a`a=== $title ===" -ForegroundColor Green
            Write-Host $msg
            break
        } catch {
            Write-Host "Parsing error, retrying..." -ForegroundColor Yellow
        }
    }
    
    # Check intermediate progress
    $progress = ssh $remoteUser@$remoteHost "tail -1 ${genDir}/_w0/worker.log 2>/dev/null || echo 'waiting...'" 2>$null
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $progress"
    
    Start-Sleep -Seconds $pollSeconds
}