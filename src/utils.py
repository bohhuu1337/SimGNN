"""Утилиты обработки данных."""

import json
import math
from texttable import Texttable

def tab_printer(args):
    """
    Функция для вывода логов в красивом табличном формате.
    :param args: Параметры, используемые для модели.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Параметр", "Значение"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Чтение json-файла с парой графов.
    :param path: Путь к файлу JSON.
    :return data: Словарь с данными.
    """
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
    return data

def calculate_loss(prediction, target):
    """
    Вычисление квадратичной потери для нормализованного GED.
    :param prediction: Предсказанное значение логарифма GED.
    :param target: Фактический логарифм GED.
    :return score: Квадратичная ошибка.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Вычисление нормализованного GED для пары графов.
    :param data: Таблица данных.
    :return norm_ged: Нормализованный показатель GED.
    """
    norm_ged = data["ged"] / max(len(data["labels_1"]), len(data["labels_2"]))

    return norm_ged
