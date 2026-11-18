@echo off
title ResumeNova - AI Resume Analyzer
echo ========================================================
echo Starting ResumeNova AI Resume Analyzer
echo ========================================================
echo.
echo [1/2] Launching Flask Backend on http://localhost:5000 ...
start "ResumeNova Backend (Port 5000)" cmd /k "cd backend && python app.py"

timeout /t 3 /nobreak >nul

echo [2/2] Launching React Frontend on http://localhost:3000 ...
start "ResumeNova Frontend (Port 3000)" cmd /k "cd frontend && npm start"

echo.
echo ========================================================
echo App is starting in two separate windows:
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:5000
echo.
echo Demo Accounts:
echo - Admin: chethang387 / 12345678
echo - User:  user / user123
echo ========================================================
