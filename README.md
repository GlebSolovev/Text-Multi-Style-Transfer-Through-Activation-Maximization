# Text-Multi-Style-Transfer-Through-Activation-Maximization

## О проекте

_Презентация:_ https://docs.google.com/presentation/d/1p8rDXqzizx0ZXpz-hTmolJyTv0tgvYGn-KnwRIGP2Q8/edit?usp=sharing

## Запуск

Для запуска проекта необходимы библиотеки, указанные в файле `requirements.txt`.
Установить их можно, например, с помощью команды:

```bash
python3 -m pip install -r requirements.txt
```

Для запуска тестового примера достаточно, находясь в корневой папке проекта, ввести команду:

```bash
python3 main.py
```

## Зависимости проекта

Классификаторы стилей взяты и адаптированы из
проекта [Style Transfer Through Back-Translation](https://github.com/shrimai/Style-Transfer-Through-Back-Translation).
А именно, пакет `onmt`, чекпоинты моделей в директории `checkpoints` и части алгоритма запуска в `sttbt_classifier.py`.
