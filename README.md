# Sentiment Analysis с BERT

## Описание
Проект по анализу тональности текстов с использованием трансформеров. Сравниваются baseline-модель (LogisticRegression на BERT-эмбеддингах) и fine-tuned BERT для классификации отзывов IMDB на положительные/отрицательные.

## Структура
- `trans.ipynb` — основной ноутбук с полным пайплайном (токенизация, эмбеддинги, обучение, оценка)
- `fine_tuned_model/` — обученная модель BERT для классификации
- `baseline_model.pkl` — сохранённая baseline-модель (LogisticRegression)
- `baseline_vectorizer.pkl` — сохранённый StandardScaler для baseline
- `app.py` — Gradio приложение
- `api.py` — FastAPI альтернатива
- `error_analysis.txt` — анализ ошибок
- `comparison_results.txt` — сравнение моделей
- `confusion_matrix_finetuned.png` — confusion matrix fine-tuned модели
- `data/imdb_small.csv` — подмножество датасета IMDB (2000 примеров)

## Запуск демо

### Вариант с Gradio:
```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

### Вариант с FastAPI:
```bash
pip install fastapi uvicorn transformers torch
uvicorn api:app --reload
```
API будет доступно на http://127.0.0.1:8000

Пример запроса:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

## Результаты

### Fine-tuned модель:
- F1 (macro): 0.7929
- Accuracy: 0.7950

### Baseline модель:
- F1 (macro): 0.7624
- Accuracy: 0.7625

Улучшение: 4.00%

## Использование в коде
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')

inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

## Требования
- Python 3.8+
- transformers
- torch
- scikit-learn
- pandas
- datasets
- gradio (для демо)
- fastapi + uvicorn (для API)
