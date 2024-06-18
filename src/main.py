"""SimGNN runner."""

from utils import tab_printer
from utils import process_pair
from simgnn import SimGNNTrainer
from param_parser import parameter_parser
import glob

def main():
    """
    Анализ параметров командной строки, чтение данных.
Обучение и оценка модели SimGNN.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    if args.load_path:
        trainer.load()
    else:
        trainer.fit()

    trainer.score()
    if args.save_path:
        trainer.save()

if __name__ == "__main__":
    main()
