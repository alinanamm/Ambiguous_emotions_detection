# Ambiguous_emotions_detection

---

## Данные

**GoEmotions** - датасет из 58 000 Reddit-комментариев, размеченных по 27 эмоциям + neutral.

Soft-labels построены из raw-версии датасета на основе долей голосов аннотаторов.

---

## Эксперименты:

| # | Разметка            | Модификация                                       |
|---|---------------------|---------------------------------------------------|
| 1 | one-hot, soft-label | TACO baseline                                     | 
| 2 | one-hot             | focal BCE с весами по обратной частоте класса     | 
| 3 | soft-label          | weighted BCE с весами согласованности аннотаторов |
| 4 | one-hot, soft-label | WeightedRandomSampler                             | 
| 5 | one-hot, soft-label | adaptive threshold                                |

## Стек

| Библиотека | Назначение |
|-----------|-----------|
| PyTorch | реализация нейросетевых моделей |
| HuggingFace Transformers | загрузка RoBERTa, токенизация |
| HuggingFace Datasets | загрузка GoEmotions |
| Pandas, NumPy, Polars | работа с данными |
| Matplotlib, Seaborn | визуализация |

