# Адаптация языковых моделей к распознаванию неоднозначных и противоречивых эмоциональных состояний в текстовых высказываниях

НИР, 3 курс, ПАДИИ НИУ ВШЭ Санкт-Петербург.

Научный руководитель: к.т.н. Рюмин Дмитрий Александрович, доцент.

## О проекте

Адаптация архитектуры TACO (Triple-View framework, ACL 2025) для полной многометочной классификации эмоций на корпусе GoEmotions: все 28 классов (включая neutral), весь датасет, без отбрасывания multi-label примеров — в отличие от большинства существующих решений, которые упрощают задачу до single-label без neutral.

Основной фокус — сравнение **one-hot** и **soft-label** разметки как способа представления эмоциональной неопределённости, и оценка вклада каждого архитектурного улучшения в качество классификации.

## Структура репозитория

Репозиторий разделён на две ветки по типу работы:

### [`EDA`](../../tree/EDA) — разведочный анализ корпуса

Анализ распределения эмоций, co-occurrence паттернов между метками, степени согласия аннотаторов, длины текстов и других характеристик GoEmotions, предшествующий выбору архитектуры и стратегии разметки.

### [`experiments`](../../tree/experiments) — эксперименты с моделью

Пошаговая адаптация TACO к multi-label постановке задачи, отдельно для one-hot и soft-label разметки:

01_baseline.py / 01_baseline_sl.py            — TACO baseline (one-hot / soft-label)

02_focal_bce.py / 02_focal_bce_sl.py           — + focal BCE с весами по обратной частоте класса

03_weighted_sampling.py / 03_weighted_sampling_sl.py  — + WeightedRandomSampler

04_adaptive_threshold.py / 04_adaptive_threshold_sl.py — + adaptive threshold (финал)

## Краткие итоговые результаты

| Модификация | Macro-F1 (one-hot) | Macro-F1 (soft-label) |
|---|---|---|
| baseline | 0.23 | 0.49 |
| + weighted BCE | **0.43** | **0.51** |
| + weighted sampling | 0.21 | 0.48 |
| + adaptive threshold | 0.42 | 0.41 |

Для сравнения, существующие решения на GoEmotions:

| Метод | Macro-F1 | Постановка задачи |
|---|---|---|
| Baseline 2020 (BERT-base) | 0.46 | 28 классов, весь корпус — та же постановка, что у нас |
| LCL (ELECTRA, 2021) | 0.63 | 27 классов без neutral, однометочные примеры |
| Llama-2 + MoE (2024) | 0.587 | 18 классов (теория Плутчика) |
| TACO original (RoBERTa, 2025) | 0.5823 | 27 классов без neutral, однометочные примеры |

**Главный результат:** soft-label с weighted BCE (0.51) обгоняет литературный Baseline 2020 (0.46), решая при этом ту же по сложности задачу — полный multi-label корпус, 28 классов.

## Используемые источники

- Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
- Gong et al. (2025). TACO: A Triple-View Framework for Fine-Grained Emotion Classification with Clustering-Guided Contrastive Learning. ACL 2025.
- Suresh & Ong (2021). Not All Negatives are Equal: Label-Aware Contrastive Loss for Fine-grained Text Classification. EMNLP 2021.
- Lim & Cheong (2024). Integrating Plutchik's Theory with Mixture of Experts for Enhancing Emotion Classification. EMNLP 2024.

Полный список источников — в презентации защиты.
