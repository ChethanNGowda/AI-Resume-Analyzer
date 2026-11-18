Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting ResumeNova AI Resume Analyzer" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$root = $PSScriptRoot

Write-Host "[1/2] Launching Flask Backend on http://localhost:5000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; Write-Host 'ResumeNova Backend running on port 5000...' -ForegroundColor Green; python app.py"

Start-Sleep -Seconds 2

Write-Host "[2/2] Launching React Frontend on http://localhost:3000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; Write-Host 'ResumeNova Frontend running on port 3000...' -ForegroundColor Green; npm start"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "Both servers launched in separate PowerShell windows!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "Backend:  http://localhost:5000" -ForegroundColor White
Write-Host "Demo:     chethang387 / 12345678 or user / user123" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Green
