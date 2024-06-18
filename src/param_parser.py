"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
   Метод для разбора параметров командной строки.
Параметры по умолчанию обеспечивают высокопроизводительную модель без поиска по сетке.
    """
    parser = argparse.ArgumentParser(description="Запуск SimGNN.")

    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="./dataset/train/",
	                help="Папка с JSON-файлами пар графов для обучения.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="./dataset/test/",
	                help="Папка с JSON-файлами пар графов для тестирования.")

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
	                help="Количество эпох обучения. По умолчанию 5.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
	                help="Фильтры (нейроны) в 1-м сверточном слое. По умолчанию 128.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
	                help="Фильтры (нейроны) во 2-м сверточном слое. По умолчанию 64.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
	                help="Фильтры (нейроны) в 3-м сверточном слое. По умолчанию 32.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Нейроны в слое тензорной сети. По умолчанию 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
	                help="Нейроны в узком слое. По умолчанию 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
	                help="Количество пар графов в пакете. По умолчанию 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Количество корзин для оценки схожести. По умолчанию 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Вероятность исключения нейронов при обучении. По умолчанию 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Скорость обучения. По умолчанию 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Вес уменьшения для оптимизатора Adam. По умолчанию 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.set_defaults(histogram=False)

    parser.add_argument("--save-path",
                        type=str,
                        default=None,
                        help="Куда сохранить обученную модель")

    parser.add_argument("--load-path",
                        type=str,
                        default=None,
                        help="Загрузить предварительно обученную модель")
    return parser.parse_args()

