[README.md](https://github.com/user-attachments/files/32356589/README.md)

# Классификация твитов о катастрофах

## Описание проекта

Цель проекта — решить задачу бинарной классификации твитов: определить, относится ли сообщение к реальной катастрофе (`target = 1`) или нет (`target = 0`).

В рамках проекта были последовательно рассмотрены три подхода:

1. Классический NLP: preprocessing + TF-IDF + классические ML-модели
2. Предобученные transformer embeddings + классический ML
3. Fine-tuning предобученного Transformer

Такой подход позволяет сравнить, насколько качество улучшается при переходе от статистических текстовых признаков к contextual embeddings и полноценному fine-tuning трансформера.

---

## Датасет

Использовался датасет Kaggle **Natural Language Processing with Disaster Tweets**.

Основные поля:

- `id` — идентификатор твита
- `keyword` — ключевое слово
- `location` — указанная пользователем локация
- `text` — текст твита
- `target` — целевой класс

В текущей версии проекта для классификации использовалось только поле `text`.

Данные были разделены на train и validation с помощью stratified split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Одинаковый validation split использовался во всех экспериментах для корректного сравнения моделей.

---

## EDA

В ходе разведочного анализа были изучены:

- распределение целевого класса
- длина твитов в символах
- длина твитов в словах
- наиболее частые слова
- URL
- mentions
- stopwords
- шумовые токены

Средняя длина твита составляла около 101 символа и примерно 15 слов.

Было замечено, что тексты содержат заметное количество URL, mentions, HTML-последовательностей, stopwords и пунктуационного шума.

---

## Классический NLP

### Preprocessing

Было протестировано несколько вариантов preprocessing:

- исходный текст без обработки
- удаление URL и mentions
- удаление stopwords
- lemmatization
- stemming

Базовая очистка:

```python
def clear_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text
```

### TF-IDF + Logistic Regression

Для векторизации использовался `TfidfVectorizer`, а основной классической моделью была `LogisticRegression`.

Результаты preprocessing-экспериментов:

| Preprocessing | F1 |
|---|---:|
| Raw text | 0.7512 |
| URL + mentions cleaning | 0.7525 |
| Stopwords removal | 0.7439 |
| Lemmatization | **0.7557** |
| Stemming | 0.7545 |

Лучший результат классического подхода: **F1 ≈ 0.756**.

### Gradient Boosting

Также был протестирован `GradientBoostingClassifier`.

Результат: **F1 ≈ 0.423**.

Качество оказалось значительно хуже Logistic Regression. Для высокоразмерных разреженных TF-IDF-признаков линейные модели в этой задаче подошли заметно лучше.

---

## Анализ ошибок

После обучения Logistic Regression был проведён error analysis.

Рассматривались:

- False Positive с высокой уверенностью
- False Negative с высокой уверенностью
- True Positive около decision threshold
- True Negative около decision threshold

Также анализировался вклад отдельных слов в предсказание Logistic Regression.

Для каждого признака:

```text
contribution = TF-IDF value × Logistic Regression weight
```

Полный logit модели:

```text
z = intercept + Σ(x_i * w_i)
```

Вероятность:

```text
p = sigmoid(z)
```

Этот анализ хорошо показал ограничение TF-IDF: модель использует статистические корреляции между отдельными словами и классом, но практически не понимает контекст предложения.

---

## Transformer Embeddings + Logistic Regression

Следующим этапом было использование предобученной embedding-модели:

**Tarka-AIR/Tarka-Embedding-150M-V1**

### Почему была выбрана эта модель

Основные причины выбора:

- хорошие результаты в MTEB и смежных benchmark-задачах по representation/classification
- размер около 150M параметров
- совместимость с Sentence Transformers
- возможность напрямую получать один dense embedding для всего текста
- хорошая применимость к коротким текстам
- разумный баланс между качеством и вычислительной стоимостью

Для небольшого учебного датасета использование намного более крупной embedding-модели было бы избыточным.

### Схема

```text
tweet
↓
Transformer
↓
token hidden states
↓
pooling
↓
768-dimensional sentence embedding
```

На выходе для N твитов получается матрица вида:

```text
(N, 768)
```

После этого embeddings подавались в Logistic Regression.

### Результаты

- Raw text: **F1 = 0.7871**
- После удаления URL и mentions: **F1 = 0.7868**

Дополнительный preprocessing практически не улучшил качество, поэтому для transformer embeddings был сохранён исходный текст.

---

## Fine-tuning DistilBERT

На последнем этапе был выполнен полноценный fine-tuning модели:

**distilbert/distilbert-base-uncased**

### Почему был выбран DistilBERT

Причины выбора:

- около 67M параметров
- заметно легче BERT-base
- быстрее обучается
- требует меньше GPU memory
- сохраняет большую часть качества BERT
- хорошо подходит для коротких английских текстов
- полностью поддерживается Hugging Face Transformers

Для этой задачи использование намного более крупной модели было бы неоправданно по вычислительной стоимости.

### Почему использовался base checkpoint

Рассматривался вариант:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

Но этот checkpoint уже был fine-tuned на sentiment analysis.

Поэтому был выбран именно:

```text
distilbert-base-uncased
```

Это позволяет адаптировать pretrained language model непосредственно под текущую задачу классификации Disaster Tweets, не начиная с весов, уже смещённых под другую downstream-задачу.

---

## Tokenization

Использовался `AutoTokenizer`.

Для каждого текста формировались:

- `input_ids`
- `attention_mask`

Максимальная длина:

```python
max_length=128
```

Для коротких твитов этого достаточно.

---

## Параметры fine-tuning

Основные параметры:

```python
num_train_epochs=3
per_device_train_batch_size=16
learning_rate=2e-5
weight_decay=0.01
```

Также использовались:

```python
load_best_model_at_end=True
metric_for_best_model="f1"
greater_is_better=True
```

Это позволяет после обучения автоматически загрузить checkpoint с максимальным F1 на validation.

---

## Результаты

Итоговое сравнение подходов:

| Модель | Представление текста | F1 |
|---|---|---:|
| Logistic Regression | TF-IDF | ~0.756 |
| Logistic Regression | Tarka Transformer Embeddings | ~0.787 |
| Fine-tuned DistilBERT | Transformer | **~0.808** |

---

## Основные выводы

Качество росло последовательно:

```text
TF-IDF
↓
Transformer embeddings
↓
Transformer fine-tuning
```

### TF-IDF

Плюсы:

- быстро
- просто
- интерпретируемо
- дёшево по вычислениям

Минусы:

- плохо понимает контекст
- почти не учитывает семантику
- чувствителен к поверхностной статистике отдельных слов

### Transformer Embeddings

Использование Tarka embeddings дало заметный прирост качества.

Преимущество подхода в том, что Transformer уже кодирует семантику и контекст, а поверх этих представлений можно быстро обучить обычную Logistic Regression.

Это хороший компромисс между качеством и стоимостью обучения.

### Fine-tuned DistilBERT

Fine-tuning показал лучший результат.

Причина в том, что модель смогла адаптировать свои внутренние representation непосредственно под задачу Disaster Tweet Classification.

В случае frozen embeddings представление текста фиксировано, а при fine-tuning меняются веса всей модели под конкретную downstream-задачу.

---

## Overfitting и выбор checkpoint

Во время обучения наблюдалось начало переобучения:

- training loss продолжал уменьшаться
- validation loss после определённой эпохи начинал расти
- F1 переставал улучшаться

Поэтому использовался выбор лучшего checkpoint по validation F1, а не просто последняя эпоха.

---

## Ограничения проекта

В текущей версии:

- не использовались поля `keyword` и `location`
- не проводился полноценный hyperparameter tuning
- не использовалась cross-validation
- не проводился отдельный Kaggle test submission
- сравнение Transformer-моделей ограничено DistilBERT
- threshold tuning оставлен как возможное дальнейшее улучшение

---

## Возможные улучшения

В дальнейшем можно протестировать:

- RoBERTa
- DeBERTa
- ModernBERT
- class weights
- threshold tuning
- learning rate scheduler
- early stopping
- hyperparameter optimization
- использование `keyword`
- использование `location`
- ensemble нескольких моделей
- cross-validation
- дополнительный error analysis Transformer-модели

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- spaCy
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Matplotlib

---

## Рекомендуемая структура проекта

```text
project/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── classic_ml.py
│   ├── embeddings.py
│   └── finetune_distilbert.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Reproducibility

Для честного сравнения всех подходов использовался один и тот же train/validation split с:

```python
random_state=42
stratify=y
```

Векторизаторы и модели обучались только на train-части, а validation использовался только для оценки качества.

---

## Итог

В проекте были сравнены три уровня NLP-подходов:

1. статистическое представление текста
2. предобученные contextual embeddings
3. end-to-end Transformer fine-tuning

Итоговая динамика качества:

```text
TF-IDF + Logistic Regression → ~0.756
Transformer Embeddings + Logistic Regression → ~0.787
Fine-tuned DistilBERT → ~0.808
```

Лучший результат показал fine-tuned DistilBERT.

При этом классический TF-IDF остаётся сильным и дешёвым baseline, pretrained embeddings дают хороший компромисс между качеством и скоростью, а fine-tuning Transformer обеспечивает максимальное качество ценой более высокой вычислительной стоимости.
