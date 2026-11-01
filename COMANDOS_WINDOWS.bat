@echo off
chcp 65001 >nul
echo ===============================
echo VISION 360 - FASE II
 echo ===============================
docker compose version || (echo Docker Desktop no disponible. & pause & exit /b 1)
docker compose up --build -d || (echo No fue posible construir los contenedores. & pause & exit /b 1)
echo.
echo Aplicacion: http://localhost:8080
echo API:        http://localhost:8080/api/docs
start http://localhost:8080
docker compose ps
pause
