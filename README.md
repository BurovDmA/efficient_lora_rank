# Efficient LoRA Rank Allocation

Код и ноутбуки для экспериментов по эффективному распределению ранга в LoRA-адаптерах.

В репозитории собраны:

- обучение базовых LoRA, L1RA и AdaLoRA адаптеров;
- warm-start переход из L1RA/AdaLoRA в обычную LoRA;
- анализ effective rank и отношения `|ΔW|_F / |W|_F`;
- эксперименты с усреднением LoRA-адаптеров.

## Структура

- `src/` - основной код обучения, загрузки данных, моделей и warm-start.
- `src/l1ra/` - реализация L1RA-слоев и конфигурации.
- `scripts/` - отдельные скрипты для построения графиков и heatmap.
- `notebooks/lora_baselines/` - ноутбуки для baseline-экспериментов и анализа рангов.
- `notebooks/lora_warm_start/` - ноутбуки для перехода L1RA/AdaLoRA -> LoRA.
- `notebooks/lora_soups/` - ноутбук с экспериментом LoRA soup.
