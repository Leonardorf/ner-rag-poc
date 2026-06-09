# Activar entorno virtual
.\venv\Scripts\Activate.ps1

# Proxy corporativo (10.20.1.194:3128) con certificado autofirmado
$env:HTTPS_PROXY = "http://10.20.1.194:3128"
$env:HTTP_PROXY  = "http://10.20.1.194:3128"
$env:NO_PROXY    = "localhost,127.0.0.1"

# Deshabilitar verificacion SSL para HuggingFace (cert corporativo autofirmado)
$env:HF_HUB_DISABLE_SSL_VERIFICATION = "1"

# Obtener IP local para compartir en la red
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress

Write-Host "Entorno listo."
Write-Host ""
Write-Host "---- USO LOCAL (solo esta maquina) ----"
Write-Host "  streamlit run src/app.py"
Write-Host "  Acceso: http://localhost:8501"
Write-Host ""
Write-Host "---- USO EN RED (otras personas de la red) ----"
Write-Host "  streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501"
Write-Host "  Compartir esta direccion: http://${ip}:8501"
Write-Host "  NOTA: requiere que el puerto 8501 este habilitado en el firewall"
Write-Host ""
Write-Host "Para ejecutar el pipeline por consola:"
Write-Host "  cd src"
Write-Host "  python pipeline_rag_ner_judicial.py"
