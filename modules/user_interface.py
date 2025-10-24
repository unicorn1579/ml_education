import argparse
from modules.common import InputParams


def process_cli() -> InputParams:
    """
    Пользовательский интерфейс

    Returns:
    --------
    InputParams
        Входные параметры
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--directory',
        help='Путь к директории с файлами, содержащими данные в виде временного ряда',
        default='input'
    )
    parser.add_argument(
        '-fn', '--files-names',
        nargs='+',
        help='Список имён файлов, содержащих данные в виде временного ряда',
        default=None
    )
    parser.add_argument(
        '-tg', '--tickets-group',
        help='Набор названий тикетов, совпадающих с названием файлов с данными в виде временного ряда',
        default=None
    )
    parser.add_argument(
        '-tc', '--time-column',
        help='Поле набора данных, содержащее временные метки',
        default='date'
    )
    parser.add_argument(
        '-tf', '--target-frequency',
        help='Целевая гранулярность',
        default='H'
    )
    parser.add_argument(
        '-fm', '--fill-method',
        help='Способ заполнения пропусков',
        default='rolling'
    )
    parser.add_argument(
        '-am', '--anomaly-method',
        help='Способ обработки выбросов',
        default='rolling'
    )
    parser.add_argument(
        '-rw', '--rolling-window',
        help='Величина скользящего окна',
        default=2,
        type=int
    )
    args = parser.parse_args()

    params = InputParams(
        directory=args.directory,
        files_names=args.files_names,
        tickets_group=args.tickets_group,
        time_column=args.time_column,
        target_frequency = args.target_frequency,
        fill_method=args.fill_method,
        anomaly_method=args.anomaly_method,
        rolling_window=args.rolling_window
    )

    return params
