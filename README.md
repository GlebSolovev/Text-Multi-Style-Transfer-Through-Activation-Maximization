# Text-Multi-Style-Transfer-Through-Activation-Maximization

## О проекте

_Презентация:_ https://docs.google.com/presentation/d/1p8rDXqzizx0ZXpz-hTmolJyTv0tgvYGn-KnwRIGP2Q8/edit?usp=sharing

## Запуск

Для запуска проекта необходимы библиотеки, указанные в файле `requirements.txt`.
Установить их можно, например, с помощью команды:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git
```

Для запуска тестового примера достаточно, находясь в корневой папке проекта, ввести команду:

```bash
python3 main.py
```

Для запуска трансформеров-baseline-ов необходимо перейти в папку `baseline_transformers` и ввести команду:
```bash
python3 main.py
```

## Зависимости проекта

### Классификаторы стилей

Классификаторы стилей настроения, политических взглядов и гендера взяты и адаптированы из
проекта [Style Transfer Through Back-Translation](https://github.com/shrimai/Style-Transfer-Through-Back-Translation).
А именно, пакет `onmt`, чекпоинты моделей в директории `checkpoints` и части алгоритма запуска в `sttbt_classifier.py`.

Классификатор формальности взят и адаптирован из
проекта [xlmr_formality_classifier](https://huggingface.co/SkolkovoInstitute/xlmr_formality_classifier), а именно,
предобученная модель в файле `formality_classifier.py`.

### Базовые подходы &mdash; трансформеры стилей

Трансформер в формальный-неформальный стили: проект [Styleformer](https://github.com/PrithivirajDamodaran/Styleformer).

Трансформер в позитивный-негативный стили:
проект [How Positive Are You: Text Style Transfer using Adaptive Style Embedding](https://github.com/kinggodhj/How-Positive-Are-You-Text-Style-Transfer-using-Adaptive-Style-Embedding).
А именно, файлы в пакете `hpay.internal` и адаптированный код в `sentiment_transformer.py`.
