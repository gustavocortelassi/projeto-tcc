@echo off
setlocal enabledelayedexpansion

if not exist logs mkdir logs

for %%m in (efficientnetv2 vit mobilevit resnet50 swin) do (
    echo.
    echo ========================================
    echo   Treinando: %%m
    echo   Log: logs\%%m.log
    echo ========================================
    echo.
    python -u treinar_%%m.py > logs\%%m.log 2>&1
    if errorlevel 1 (
        echo [AVISO] %%m terminou com erro - seguindo para o proximo modelo. Veja logs\%%m.log
    ) else (
        echo [OK] %%m concluido.
    )
)

echo.
echo Todos os modelos processados. Confira outputs\results\resultado_*.json
endlocal
