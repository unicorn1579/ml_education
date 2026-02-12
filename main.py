import json
from typing import List

from joblib import Parallel, delayed

from time_series import TimeSeries
from user_interface import InputParams, process_cli


def get_files_names(params: InputParams) -> List[str]:
    """
    Возвращает список имён файлов с данными
    """
    if params.tickets_group:
        with open('configuration.json', 'r') as file:
            configuration = json.load(file)
        return configuration['tickets_groups'][params.tickets_group]
    else:
        return params.files_names

def process_ts(file_name: str, time_series_params: dict) -> TimeSeries:
    """
    Обработка одного временного ряда
    time_series_params — словарь параметров для обработки временного ряда
    """
    ts = TimeSeries(file_name=file_name, **time_series_params)
    ts.process()
    return ts

def main() -> None:
    params: InputParams = process_cli()

    files_names = get_files_names(params)

    time_series_params = dict(
        directory=params.directory,
        file_extension=params.file_extension,
        time_column=params.time_column,
        processed_columns=params.processed_columns,
        target_frequency=params.target_frequency,
        frequency_method=params.frequency_method,
        missing_method=params.missing_method,
        anomaly_method=params.anomaly_method,
        rolling_window=params.rolling_window,
        z_threshold=params.z_threshold,
        decompose_model=params.decompose_model,
        decompose_period=params.decompose_period,
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        random_state=params.random_state,
        eval_metric=params.eval_metric,
        early_stopping_rounds=params.early_stopping_rounds
    )

    # Параллельная обработка временных рядов
    time_series: List[TimeSeries] = Parallel(n_jobs=-1, backend='threading', verbose=10)(
        delayed(process_ts)(file_name, time_series_params)
        for file_name in files_names
    )

    # После обработки визуализируем
    for ts in time_series:
        ts.show()

if __name__ == '__main__':
    main()

"""
Example 1:
python main.py -d data -tg NDXT_30
Example 2:
python main.py -d data -fn A AA
"""
