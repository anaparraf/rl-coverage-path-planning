# APS08 - CPP com observação parcial (5x5/10x10)

Este documento descreve a estratégia implementada, como treinar/testar e como gerar gráficos, GIF e tabela automática de métricas.

## ✅ Estratégia implementada
- **Observação parcial 5x5 (agente no centro)**: parametrizado via `--obs-window 5`.
- **CNN + MLP (MultiInputPolicy com feature extractor customizado)** para capturar padrões espaciais locais.
- **Transfer learning (5x5 → 10x10)** usando `PPO.load` no modo `curriculum`.
- **Hiperparâmetros ajustáveis** por CLI (`gamma`, `n_steps`, `ent_coef`, etc.).

## 🔧 Como treinar
### 1) Treinar no 5x5
```powershell
python train_grid_world_cpp.py train 5 3 200 1000000 --obs-window 5
```

### 2) Continuar no 10x10 (transfer learning)
```powershell
python train_grid_world_cpp.py curriculum 10 12 400 1000000 --model-path data/SEU_MODELO_5x5.zip --obs-window 5
```

> Se preferir treinar do zero no 10x10, use `train` ao invés de `curriculum`.

## ✅ Como testar (gera CSV de métricas)
```powershell
python train_grid_world_cpp.py test 5 3 --model-path data/SEU_MODELO_5x5.zip --obs-window 5
python train_grid_world_cpp.py test 10 12 --model-path data/SEU_MODELO_10x10.zip --obs-window 5
```

Ao final do teste, um arquivo CSV é salvo em `results/` com as métricas por episódio.

## 📊 Gerar gráficos
```powershell
python utils/plot_metrics.py --log-csv log/SEU_LOG/progress.csv --metrics-csv results/SEU_METRICS.csv --out-dir results
```

Gera:
- `learning_curve.png`
- `coverage_per_episode.png`
- `full_coverage_rate.png`

## 📋 Gerar tabela automática
```powershell
python utils/metrics_table.py --metrics-csv results/SEU_METRICS.csv --out-path results/metrics_summary.md
```

## 🎞️ Gerar GIF do teste
```powershell
python utils/generate_gif.py --model-path data/SEU_MODELO_10x10.zip --dim 10 --obstacles 12 --max-steps 400 --obs-window 5 --out-path results/cpp_run.gif
```

## 🗂️ Onde colocar os resultados
- Modelos: `data/`
- Logs: `log/`
- Gráficos, tabela e GIF: `results/`

## 📝 Dica para o relatório
Inclua:
- Curva de aprendizado (reward médio vs timesteps)
- Cobertura média e full coverage rate
- GIF do comportamento do agente
- Tabela automática de métricas

## 📄 Templates do relatório
- `REPORT_TEMPLATE.md` (Markdown)
- `REPORT_TEMPLATE.tex` (LaTeX/Overleaf)
