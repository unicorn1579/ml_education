import argparse
from typing import NamedTuple, List


class InputParams(NamedTuple):
    """
    Входные параметры
    """
    directory: str = None
    files_names: List[str] = None
    tickets_group: str = None
    file_extension: str = None
    time_column: str = None
    processed_columns: List[str] = None
    target_frequency: str = None
    frequency_method: str = None
    missing_method: str = None
    anomaly_method: str = None
    rolling_window: int = None
    z_threshold: float = None
    decompose_model: str = None
    decompose_period: str = None
    n_estimators: int = None
    max_depth: int = None
    learning_rate: float = None
    subsample: float = None
    colsample_bytree: float = None
    random_state: int = None
    eval_metric: str = None
    early_stopping_rounds: int = None

def process_cli() -> InputParams:
    """
    Параметры пользовательского интерфейса
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--directory',
        help='Директория размещения файла с данными',
        default='data'
    )
    parser.add_argument(
        '-fn', '--files-names',
        nargs='+',
        help='Список имён файлов, содержащих данные',
        default=None
    )
    parser.add_argument(
        '-tg', '--tickets-group',
        help='Набор названий тикетов, совпадающих с названием файлов с данными',
        default=None
    )
    parser.add_argument(
        '-fe', '--file-extension',
        help='Расширение файла с данными',
        default='csv'
    )
    parser.add_argument(
        '-tc', '--time-column',
        help='Имя столбца с временем',
        default='date'
    )
    parser.add_argument(
        '-pc', '--processed-columns',
        nargs='+',
        help='Набор обрабатываемых полей',
        default=None
    )
    parser.add_argument(
        '-tf', '--target-frequency',
        help='Целевая гранулярность',
        default='H'
    )
    parser.add_argument(
        '-fm', '--frequency-method',
        help='Способ обработки значений при изменении гранулярности',
        default='mean'
    )
    parser.add_argument(
        '-ms', '--missing-method',
        help='Метод заполнения пропусков',
        default='rolling'
    )
    parser.add_argument(
        '-am', '--anomaly-method',
        help='Метод исправления выбросов',
        default='rolling'
    )
    parser.add_argument(
        '-rw', '--rolling-window',
        help='Окно для скользящего среднего',
        default=2,
        type=int
    )
    parser.add_argument(
        '-zt', '--z-threshold',
        help='Порог z-score для выявления выбросов',
        default=3.0,
        type=float
    )
    parser.add_argument(
        '-dm', '--decompose-model',
        help='Способ разложения временного ряда',
        default='additive'
    )
    parser.add_argument(
        '-dp', '--decompose-period',
        help='Период временного ряда',
        default=None,
        type=int
    )

    args = parser.parse_args()

    params = InputParams(
        directory=args.directory,
        files_names=args.files_names,
        tickets_group=args.tickets_group,
        file_extension=args.file_extension,
        time_column=args.time_column,
        processed_columns=args.processed_columns,
        target_frequency=args.target_frequency,
        frequency_method=args.frequency_method,
        missing_method=args.missing_method,
        anomaly_method=args.anomaly_method,
        rolling_window=args.rolling_window,
        z_threshold=args.z_threshold,
        decompose_model=args.decompose_model,
        decompose_period=args.decompose_period,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        eval_metric=args.eval_metric,
        early_stopping_rounds=args.early_stopping_rounds
    )

    return params
