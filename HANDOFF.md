# Handoff — AuK-Flash RU fine-tune

## Что переносить

**На трен-машину нужна только папка `finetune/`** (+ свежий клон upstream AuK и
веса — качаются на месте). Остальное в репо — локальный MLX-стек для инференса,
он понадобится позже при деплое, для обучения не нужен.

Проще всего: `git clone` этого репо на трен-бокс и работать в `finetune/`.

## Требования

- GPU ≥24 GB VRAM (L40S/48GB — идеально: дефолтный `frames_threshold=2700`).
  На 24 GB: `frames_threshold=1200 max_samples=4`.
- Linux + CUDA PyTorch. ~30 GB диска (веса ~17 GB + данные).
- Не Modal-специфично: любой бокс/ноутбук/аренда.

## Сетап (~15 мин + скачивания)

```bash
git clone https://github.com/Tencent-Hunyuan/AuK.git && cd AuK
git checkout e3828bdac1712bbf7bcb3a2a01eec6fa5168fe32
git apply /path/to/finetune/patches/0001-flash-grid-training.patch
git apply /path/to/finetune/patches/0002-cond-cache-and-speed.patch
pip install -e ".[train]" datasets soundfile
# опционально: pip install bitsandbytes flash-attn
```

Веса (~17 GB) — команды в `finetune/README.md` секция Setup
(auk_flash.safetensors + vae.safetensors + Qwen шарды 1-2; шард 3 не нужен).

## Запуск

```bash
# пробный прогон: CV-ru ~20ч, ночь на L40S, ~$10
AUK_REPO=$PWD/AuK PROFILE=trial bash finetune/pipeline.sh

# большой корпус: CV-ru + Golos + Sova ~400ч+
AUK_REPO=$PWD/AuK PROFILE=sota  bash finetune/pipeline.sh
```

`pipeline.sh` = fetch → prepare_jsonl → smoke → precompute → train.
Стадии скипаются `SKIP_FETCH=1 SKIP_PREPARE=1 ...`.

Слушать прогресс: `runs/auk_flash_ru/samples/update_N_gen.wav` каждые 200
апдейтов — при `flash_grid` синтезится сразу 4-шаговым рецептом.

## Что под капотом (важно понимать)

- **flash_grid**: обучение только в 4 дистиллированных t-точках — 4-шаговость
  сохраняется по построению. Off-label: если русский не встанет —
  `init_ckpt=auk_base` + `t_sampling=logistic_normal` (обычный CFM, 32 шага).
- **cond_cache**: thinker+VAE прогоняются один раз оффлайн → обучение без 3B-
  энкодера (−6 GB VRAM, шаг ~2-3× дешевле). Кэш = `data/cache/{train,val}`.
  Поменял jsonl/ckpt/процессор → пересчитай кэш (`smoke_test --cache`
  покажет STALE по фингерпринту).
- **Тюнится только DiT + layer fusion**; thinker и VAE заморожены.
- **EMA**: для прогонов <10k апдейтов ставь `ema_beta=0.999` — дефолтный
  0.9999 не успеет догнать веса, а деплоится именно EMA.
- **Валидация speaker-disjoint** (`--val-by-speaker`) — протечки нет.

## Деплой результата

```bash
python finetune/export_ema.py --ckpt runs/auk_flash_ru/model_last.pt \
    --out auk_flash_ru.safetensors --ref ckpts/AuK-Flash/auk_flash.safetensors
```

Дальше `convert_to_mlx.py` + `quantize_mlx.py` из этого репо → MLX-8bit
артефакт как у vanch007, или safetensors в ComfyUI-пак.

## Что проверено / что нет

| Проверено (CPU, без весов) | Не проверено |
|---|---|
| оба патча apply на `e3828bd` + compile | полный прогон на GPU (нет железа) |
| flash_grid → ровно 4 t-точки | сколько часов реально займёт тюн |
| кэш-путь: .pt→collate→forward+backward | качество русского после тюна |
| speaker-disjoint сплит, фингерпринт | fetch_dataset против живых HF-зеркал |

## Troubleshooting

- `OOM` → `frames_threshold=1200 max_samples=4`
- `cond_cache train dir not found` → прогнать `precompute_cond.py`
- `STALE cache` → кэш от другого jsonl, пересчитать
- `fetch_dataset` не нашёл колонки → печатает фактические, поправить
  `TEXT_CANDIDATES`/`audio_col`
- EMA-семплы звучат как база → `ema_beta` слишком высокий для длины прогона
