# DialogSum: суммаризация диалогов с помощью SmolLM2 и LoRA

## О проекте

В этом проекте исследуется задача автоматической суммаризации диалогов на датасете **DialogSum**.

Основная цель — сравнить:

1. простой non-ML baseline;
2. **SmolLM2-360M-Instruct** в zero-shot режиме;
3. ту же модель после **LoRA fine-tuning**.

В финальной постановке задачи модель получает не только сам диалог, но и `topic`, который определяет, на каком аспекте диалога необходимо сфокусировать summary.

```text
topic + dialogue → summary
```

Эксперимент построен как последовательное сравнение:

```text
Подготовка данных
        ↓
EDA
        ↓
Non-ML baseline
        ↓
Zero-shot SmolLM2
        ↓
BLEU / ROUGE
        ↓
LoRA fine-tuning
        ↓
Inference обученной модели
        ↓
BLEU / ROUGE
        ↓
LLM-as-a-judge
        ↓
Анализ результатов
```

---

## Датасет

Использовался датасет **DialogSum**.

Основные поля:

- `dialogue` — исходный диалог;
- `topic` — тема / аспект диалога;
- `summary` — эталонное summary;
- `id` — технический идентификатор.

В финальной постановке:

```text
input  = topic + dialogue
target = summary
```

В обучающей выборке использовалось **12 460 примеров**.

В тестовой выборке одинаковый диалог может встречаться несколько раз с разными `topic` и соответствующими им `summary`. Поэтому тестовые строки не группируются по `dialogue`: каждая строка считается отдельным примером.

Оценка проводится по схеме:

```text
prediction_i ↔ reference_i
```

то есть для каждого `topic + dialogue` используется соответствующий ему один reference summary.

---

# 1. Non-ML baseline

В качестве простого baseline использовалось объединение первого и последнего предложения диалога.

Baseline не использует `topic` и служит простой отправной точкой для сравнения с LLM.

### Результаты

| Метрика    | Non-ML |
|------------|-------:|
| BLEU       | 0.0273 |
| ROUGE-1    | 0.1505 |
| ROUGE-2    | 0.0417 |
| ROUGE-L    | 0.1298 |
| ROUGE-Lsum | 0.1299 |

Низкие значения показывают, что простая эвристика плохо решает задачу topic-aware summarization.

---

# 2. Zero-shot SmolLM2

В качестве основной модели использовалась:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

В zero-shot режиме модель получала:

```text
Topic: ...
Dialogue: ...
```

и инструкцию сформировать краткое summary с фокусом на заданном `topic`.

Inference выполнялся с использованием **vLLM**.

### Результаты Zero-shot

| Метрика    | Zero-shot |
|------------|----------:|
| BLEU       | 0.0575 |
| ROUGE-1    | 0.2568 |
| ROUGE-2    | 0.0615 |
| ROUGE-L    | 0.1958 |
| ROUGE-Lsum | 0.2154 |

Zero-shot модель заметно превосходит простой non-ML baseline.

При этом `length_ratio = 2.11`, то есть генерации модели в среднем значительно длиннее эталонных summary.

Ручная проверка также показывает, что zero-shot модель иногда:

- генерирует слишком длинные ответы;
- частично копирует диалог;
- добавляет лишние детали;
- допускает фактические ошибки.

---

# 3. LoRA fine-tuning

После zero-shot оценки модель была адаптирована под DialogSum с помощью **LoRA**.

Во время SFT вход модели формировался одинаково с zero-shot inference:

```text
topic + dialogue
```

а целевым ответом являлся:

```text
summary
```

Это позволяет сравнивать zero-shot и LoRA в одинаковой постановке задачи.

### LoRA configuration

```text
r = 16
lora_alpha = 16
lora_dropout = 0
bias = none
```

LoRA применялась к:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

### Параметры обучения

```text
Epochs:                       1
Batch size per device:       16
Gradient accumulation:       1
Effective batch size:        16

Learning rate:               2e-4
Warmup steps:                50
Weight decay:                0.01
LR scheduler:                linear

Precision:                   BF16
Max sequence length:         2048
Packing:                     False
Gradient checkpointing:      False
```

Размер train-набора:

```text
12 460 примеров
```

При batch size = 16 одна эпоха соответствует:

```text
779 training steps
```

Для обучения использовались **Unsloth**, **TRL SFTTrainer** и **PEFT**.

Validation loss снижался в течение обучения, при этом явного роста validation loss к концу эпохи не наблюдалось.

---

# 4. Результаты LoRA

После fine-tuning LoRA-адаптер был подключён к базовой модели через vLLM и использован для генерации summary на тестовой выборке.

| Метрика    | Zero-shot | LoRA |
|------------|----------:|-----:|
| BLEU       | 0.0575 | **0.1899** |
| ROUGE-1    | 0.2568 | **0.4209** |
| ROUGE-2    | 0.0615 | **0.1643** |
| ROUGE-L    | 0.1958 | **0.3406** |
| ROUGE-Lsum | 0.2154 | **0.3406** |

После LoRA наблюдается рост по всем автоматическим метрикам.

Например:

```text
BLEU:
0.0575 → 0.1899

ROUGE-1:
0.2568 → 0.4209

ROUGE-2:
0.0615 → 0.1643

ROUGE-L:
0.1958 → 0.3406
```

Таким образом, LoRA fine-tuning существенно улучшил способность небольшой SmolLM2 формировать summary, соответствующие заданному `topic`.

Ручная проверка также показывает, что модель лучше выделяет ключевые события, хотя отдельные неточности и лишние детали всё ещё встречаются.

---

# 5. LLM-as-a-judge

BLEU и ROUGE измеряют сходство с reference summary, но не полностью отражают фактическую корректность генерации.

Поэтому дополнительно используется **LLM-as-a-judge**.

Judge получает:

- `topic`;
- исходный `dialogue`;
- соответствующий `reference summary`;
- zero-shot summary;
- LoRA summary.

Оценка проводится по шкале от 1 до 5 по четырём критериям.

### Factuality

Насколько все утверждения summary подтверждаются исходным диалогом.

### Coverage

Насколько хорошо summary покрывает важную информацию, относящуюся к заданному `topic`.

### Conciseness

Насколько summary краткое и сфокусированное.

### Overall

Общая оценка качества summary с учётом заданного `topic`.

> Результаты LLM-as-a-judge необходимо обновить после повторного запуска оценки по новой topic-aware постановке.

---

# 6. Что изменилось после обучения

До обучения модель уже могла выполнять задачу суммаризации, однако качество было заметно ниже.

После LoRA:

- summaries стали ближе к эталонным;
- модель лучше учитывает заданный `topic`;
- выросло лексическое совпадение с reference;
- выросли BLEU и ROUGE;
- генерации стали лучше отражать целевой аспект диалога.

При этом дообучение не сделало модель полностью безошибочной.

В отдельных примерах всё ещё встречаются:

- пропущенная информация;
- неточные формулировки;
- отдельные фактические ошибки;
- добавление информации, которой не было в исходном диалоге.

---

# 7. Итоговое сравнение

| Модель    | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----------|-----:|--------:|--------:|--------:|
| Non-ML    | 0.0273 | 0.1505 | 0.0417 | 0.1298 |
| Zero-shot | 0.0575 | 0.2568 | 0.0615 | 0.1958 |
| **LoRA**  | **0.1899** | **0.4209** | **0.1643** | **0.3406** |

Наиболее заметное улучшение происходит после LoRA fine-tuning.

```text
Non-ML → Zero-shot → LoRA
```

Все три подхода оцениваются по одной и той же one-to-one схеме на тестовой выборке.

---

# 8. Финальный вывод

В рамках проекта был построен полный pipeline для адаптации небольшой LLM под задачу topic-aware summarization.

Модель получает:

```text
topic + dialogue
```

и должна сформировать соответствующее:

```text
summary
```

Сначала была протестирована простая эвристика, затем исходная SmolLM2-360M-Instruct в zero-shot режиме, после чего модель была дообучена с помощью LoRA.

LoRA существенно улучшила результаты:

```text
BLEU       0.0575 → 0.1899
ROUGE-1    0.2568 → 0.4209
ROUGE-2    0.0615 → 0.1643
ROUGE-L    0.1958 → 0.3406
```

Результаты показывают, что даже небольшая instruction-модель размером около 360M параметров может быть эффективно адаптирована под конкретную задачу с помощью parameter-efficient fine-tuning.

При этом автоматические метрики не гарантируют фактическую корректность, поэтому результаты дополнительно проверяются вручную и с помощью LLM-as-a-judge.

---

## Ограничения эксперимента

1. LoRA обучалась только одну эпоху.
2. Полноценный подбор гиперпараметров не проводился.
3. BLEU и ROUGE не полностью отражают factuality.
4. После fine-tuning отдельные summary всё ещё могут содержать неточности.
5. LLM-as-a-judge зависит от выбранной judge-модели и prompt.
6. Topic-aware постановка отличается от первоначальной multi-reference оценки, поэтому старые и новые метрики нельзя напрямую сравнивать.

---

## Стек

```text
Python
PyTorch
Hugging Face Transformers
Datasets
Evaluate
TRL
Unsloth
PEFT / LoRA
vLLM
Pandas
Matplotlib
```

---

## Окружения

Из-за различий в зависимостях vLLM и Unsloth использовались два изолированных Python-окружения.

```text
dialogsum
├── Unsloth
├── TRL
├── PEFT
└── training

dialogsum-vllm
├── vLLM
└── inference
```

LoRA-адаптер сохраняется на диск после обучения и затем загружается vLLM для inference.

---

## Структура проекта

```text
.
├── README.md
├── DialogSum_*.ipynb
├── judge_results.csv
└── ...
```

Большие веса модели и секретные API-ключи не должны храниться в GitHub-репозитории.

---

## Как запустить

Проект использует два окружения.

### Training environment

```bash
conda create -n dialogsum python=3.11 -y
conda activate dialogsum

pip install torch transformers datasets evaluate rouge-score nltk pandas matplotlib
pip install unsloth trl peft accelerate
```

### vLLM inference environment

```bash
conda create -n dialogsum-vllm python=3.12 -y
conda activate dialogsum-vllm

pip install --upgrade uv
uv pip install vllm --torch-backend=auto
pip install pandas evaluate rouge-score nltk
```

Если FlashInfer пытается JIT-компилировать CUDA kernel и не находит `nvcc`, перед импортом vLLM можно отключить FlashInfer sampler:

```python
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
```

После этого pipeline выполняется в следующем порядке:

```text
1. Загрузка и подготовка данных
2. EDA
3. Non-ML baseline
4. Zero-shot inference
5. BLEU / ROUGE
6. LoRA fine-tuning
7. Сохранение LoRA adapter
8. Inference обученной модели через vLLM
9. Повторная оценка BLEU / ROUGE
10. LLM-as-a-judge
11. Финальное сравнение
```

Для LLM-as-a-judge требуется API-токен. Секретные токены следует передавать через переменные окружения или `getpass` и не сохранять в notebook или GitHub.
