# Treina os modelos em sequencia.
#
# Cada modelo roda como processo separado - se um crashar de forma
# irrecuperavel, o script segue para o proximo em vez de travar tudo num
# so. Cada modelo pula sozinho o que ja esta treinado (fold ou modelo
# completo), entao rodar este script de novo depois de uma interrupcao e
# seguro: ele so retoma o que falta.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\treinar_todos.ps1

$ErrorActionPreference = 'Continue'
$modelos = @('efficientnetv2', 'vit', 'mobilevit', 'resnet50', 'swin')

New-Item -ItemType Directory -Force -Path .\logs | Out-Null

foreach ($modelo in $modelos) {
    $logPath = ".\logs\${modelo}.log"

    Write-Host ""
    Write-Host "========================================"
    Write-Host "  Treinando: $modelo"
    Write-Host "  Log: $logPath"
    Write-Host "========================================"
    Write-Host ""

    python -u ".\treinar_$modelo.py" 2>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[AVISO] $modelo terminou com codigo $LASTEXITCODE - seguindo para o proximo modelo."
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "[OK] $modelo concluido."
        Write-Host ""
    }
}

Write-Host ""
Write-Host "Todos os modelos processados. Confira outputs\results\resultado_*.json"
