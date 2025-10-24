from modules.transformer import TimeSeriesTransformer
from modules.user_interface import process_cli
from modules.common import InputParams, TimeSeries
from modules.reader import Reader

def main() -> None:
    params: InputParams = process_cli()

    reader = Reader(params=params)
    reader.run()
    time_series: TimeSeries = reader.time_series

    transformer = TimeSeriesTransformer(
        time_column=params.time_column,
        target_frequency=params.target_frequency,
        fill_method=params.fill_method,
        anomaly_method=params.anomaly_method,
        rolling_window=params.rolling_window
    )

    # Обрабатываем список DataFrame-ов, содержащих временные ряды
    time_series.data_transformed = transformer.fit_transform(time_series.data)

    time_series.show()

if __name__ == '__main__':
    main()

"""
Example 1:
python main.py -d data -tg NDXT_30
Example 2:
python main.py -d data -fn A
Example 3:
python main.py -d data -fn A AA
"""
