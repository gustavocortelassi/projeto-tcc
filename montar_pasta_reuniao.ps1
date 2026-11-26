# Organiza as imagens da reunião com o orientador numa pasta única,
# numerada em ordem, a partir do que já foi gerado em
# outputs/plots/relatorio/ (09_gerar_relatorio_visual.py).
#
# Uso (rodar de dentro de "Projeto Final"):
#   powershell -ExecutionPolicy Bypass -File .\montar_pasta_reuniao.ps1

$origem = "outputs\plots\relatorio"
$destino = "reuniao"

if (Test-Path $destino) {
    Remove-Item $destino -Recurse -Force
}
New-Item -ItemType Directory -Path $destino | Out-Null

# 1. Comparativo geral (ranking + mapa de calor por classe)
$pastaComp = Join-Path $destino "01_comparativo"
New-Item -ItemType Directory -Path $pastaComp | Out-Null
Copy-Item (Join-Path $origem "comparativo\f1_comparativo.png") $pastaComp -ErrorAction SilentlyContinue
Copy-Item (Join-Path $origem "comparativo\f1_por_classe_heatmap.png") $pastaComp -ErrorAction SilentlyContinue

# 2. Um modelo por pasta, numerado — mesma ordem da tabela do artigo
$modelos = @(
    @{ Num = "02"; Nome = "resnet50" },
    @{ Num = "03"; Nome = "efficientnetv2" },
    @{ Num = "04"; Nome = "vit" },
    @{ Num = "05"; Nome = "swin" },
    @{ Num = "06"; Nome = "mobilevit" }
)

foreach ($m in $modelos) {
    $pastaModelo = Join-Path $destino "$($m.Num)_$($m.Nome)"
    New-Item -ItemType Directory -Path $pastaModelo | Out-Null

    $origemModelo = Join-Path $origem $m.Nome
    Copy-Item (Join-Path $origemModelo "curva_treino.png") $pastaModelo -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $origemModelo "matriz_confusao.png") $pastaModelo -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $origemModelo "f1_por_classe.png") $pastaModelo -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $origemModelo "variancia_folds.png") $pastaModelo -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Pasta 'reuniao' montada:"
Get-ChildItem $destino -Recurse -Filter *.png | ForEach-Object {
    Write-Host "  $($_.FullName.Replace((Get-Location).Path + '\', ''))"
}

$faltando = Get-ChildItem $destino -Recurse -Directory | Where-Object {
    (Get-ChildItem $_.FullName -Filter *.png).Count -eq 0
}
if ($faltando) {
    Write-Host ""
    Write-Host "AVISO: pastas sem nenhuma imagem encontrada (arquivo de origem pode nao existir):"
    $faltando | ForEach-Object { Write-Host "  $($_.Name)" }
}
