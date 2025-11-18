from dataclasses import dataclass
from pathlib import Path
from typing import List
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            time_column: str = None,
            processed_columns: List[str] = None,
            target_frequency: str = None,
            frequency_method: str = None,
            missing_method: str = None,
            anomaly_method: str = None,
            rolling_window: int = None,
            z_threshold: float = None
    ):
        """
        Инициализирует объект TimeSeriesTransformer.

        Args:
            time_column (str): имя столбца с временем
            processed_columns (List[str]): набор обрабатываемых столбцов
            target_frequency (str): желаемая частота ('H', 'D', 'T' и т.д.)
            frequency_method (str): способ обработки значений при изменении гранулярности
            missing_method (str): метод заполнения пропусков
            anomaly_method (str): метод исправления выбросов
            rolling_window (int): окно для скользящего среднего
            z_threshold (float): порог z-score для выявления выбросов
        """
        self.time_column = time_column
        self.processed_columns = processed_columns
        self.target_frequency = target_frequency
        self.frequency_method = frequency_method
        self.missing_method = missing_method
        self.anomaly_method = anomaly_method
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold

        self.frequency = None

    def transform(self, data: pd.DataFrame):
        """
        Преобразовывает входной набор данных
        """
        data_transformed = self.data_prepared(data)
        data_transformed = self.time_column_transformed(data_transformed)
        data_transformed = self.missing_transformed(data_transformed)
        data_transformed = self.frequency_transformed(data_transformed)
        data_transformed = self.outliers_transformed(data_transformed)
        return data_transformed

    def data_prepared(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка исходных данных
        """
        if not self.processed_columns:
            self.processed_columns = data.select_dtypes(include='number').columns
        return data.copy()

    def time_column_transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразовывает столбец времени в тип datetime
        """
        data[self.time_column] = pd.to_datetime(data[self.time_column], errors='coerce', utc=True)
        data = data.dropna(subset=[self.time_column])
        data = data.set_index(self.time_column).sort_index()
        return data

    def missing_transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразовывает пропуски
        """
        if self.missing_method == 'mean':
            data[self.processed_columns] = data[self.processed_columns].fillna(
                data[self.processed_columns].mean()
            )
        elif self.missing_method == 'ffill':
            data[self.processed_columns] = data[self.processed_columns].ffill()
        elif self.missing_method == 'rolling':
            data[self.processed_columns] = data[self.processed_columns].fillna(
                data[self.processed_columns].rolling(self.rolling_window, min_periods=1).mean()
            )
        return data

    def frequency_transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Определяет и преобразовывает гранулярность
        """
        # Пробуем определить частоту автоматически
        self.frequency = pd.infer_freq(data.index)
        if self.frequency is None:
            # Частота не определена, можно попробовать найти "моду" разниц между датами
            deltas = data.index.to_series().diff().dropna()
            freq = deltas.mode()[0]  # наиболее часто встречающийся интервал
            # Преобразуем в строку формата pandas
            self.frequency = pd.tseries.frequencies.to_offset(freq).freqstr
            #data = data.asfreq(self.frequency)
        current_offset = to_offset(self.frequency)
        target_offset = to_offset(self.target_frequency)

        # Частота уменьшается
        if self.target_frequency and target_offset > current_offset:
            if self.frequency_method == 'mean':
                data = data[self.processed_columns].resample(self.target_frequency).mean()
            elif self.frequency_method == 'sum':
                data = data[self.processed_columns].resample(self.target_frequency).sum()
            else:
                data = data[self.processed_columns].resample(self.target_frequency).mean()
        # Частота увеличивается
        elif self.target_frequency and target_offset <= current_offset:
            res = data[self.processed_columns].resample(self.target_frequency)
            if self.frequency_method == 'ffill':
                data = res.ffill().bfill()
            elif self.frequency_method == 'bfill':
                data = res.bfill().ffill()
            else:
                data = data[self.processed_columns].resample(self.target_frequency).ffill()
        # Гарантируем отсутствие пропусков
        if data[self.processed_columns].isna().any().any():
            # fallback, если остались NaN
            data[self.processed_columns] = data[self.processed_columns].ffill().bfill()
        return data

    def outliers_transformed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразовывает выбросы
        """
        for col in self.processed_columns:
            zscore = (data[col] - data[col].mean()) / data[col].std()
            mask = np.abs(zscore) > self.z_threshold
            if mask.any():
                if self.anomaly_method == 'rolling':
                    rolling_mean = data[col].rolling(self.rolling_window, min_periods=1).mean()
                    data.loc[mask, col] = rolling_mean[mask]
                elif self.anomaly_method == 'last':
                    data.loc[mask, col] = data[col].ffill()[mask]
        return data


@dataclass
class TimeSeries:
    """
    Инициализирует объект TimeSeries.

    Args:
        directory (str): директория размещения файла с данными
        file_name (str): наименование файла с данными
        file_extension (str): расширение файла с данными
        time_column (str): имя столбца с временем
        processed_columns (List[str]): набор обрабатываемых полей

        target_frequency (str): целевая гранулярность ('H', 'D', 'T' и т.д.)
        frequency_method (str): способ обработки значений при изменении гранулярности
        missing_method (str): метод заполнения пропусков
        anomaly_method (str): метод исправления выбросов
        rolling_window (int): окно для скользящего среднего
        z_threshold (float): порог z-score для выявления выбросов

        decompose_model (str): способ разложения временного ряда
        decompose_period (str): период временного ряда

        full_file_name (str): полное наименование файла с данными
        file_path (Path): путь к файлу с данными
        data (pd.DataFrame): исходный набор данных
        data_transformed (pd.DataFrame): преобразованный набор данных
        data_decomposed (pd.DataFrame): разложенный набор данных
    """
    directory: str = None
    file_name: str = None
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
    decompose_period: int = None

    full_file_name: str = None
    file_path: Path = None
    data: pd.DataFrame = None
    data_transformed: pd.DataFrame = None
    data_decomposed:  pd.DataFrame = None


    def __post_init__(self):
        if self.file_name and self.file_extension:
            self.full_file_name = f"{self.file_name}.{self.file_extension}"
        if self.directory and self.full_file_name:
            self.file_path = Path(self.directory) / self.full_file_name

    def process(self) -> None:
        """
        Обработка временного ряда
        """
        self.read()
        self.transform()
        self.decompose()

    def read(self) -> None:
        """
        Считывает исходные данные
        """
        if not self.file_path:
            self.data = pd.DataFrame()
        elif self.file_extension == 'csv':
            self.data = pd.read_csv(self.file_path)

    def transform(self) -> None:
        """
        Обрабатывает данные временного ряда с помощью трансформера
        """
        transformer = TimeSeriesTransformer(
            time_column = self.time_column,
            processed_columns = self.processed_columns,
            target_frequency = self.target_frequency,
            frequency_method = self.frequency_method,
            missing_method = self.missing_method,
            anomaly_method = self.anomaly_method,
            rolling_window = self.rolling_window,
            z_threshold = self.z_threshold
        )
        self.data_transformed = transformer.transform(self.data)

    def decompose(self) -> None:
        """
        Разложение временных рядов на тренд, сезонность и остаток
        """
        data_decomposed = {}
        for column in self.processed_columns:
            column_decomposed = seasonal_decompose(
                self.data_transformed[column],model=self.decompose_model,period=self.decompose_period
            )
            data_decomposed[f"{column}__observed"] = column_decomposed.observed
            data_decomposed[f"{column}__trend"] = column_decomposed.trend
            data_decomposed[f"{column}__seasonal"] = column_decomposed.seasonal
            data_decomposed[f"{column}__resid"] = column_decomposed.resid
        self.data_decomposed = pd.DataFrame(data_decomposed)


    def show(self) -> None:
        """
        Визуализация временного ряда
        """
        column_n = len(self.processed_columns)
        fig, axes = plt.subplots(column_n, 1, figsize=(12, 4 * column_n), sharex=True)
        if column_n == 1:
            axes = [axes]

        for ax, column in zip(axes, self.processed_columns):
            ax.plot(
                self.data_transformed.index,
                self.data_transformed[column],
                label="Наблюдаемые данные",
                color="orange",
                linewidth=2,
                linestyle="--"
            )
            ax.plot(
                self.data_transformed.index,
                self.data_decomposed[f"{column}__trend"],
                label="Тренд",
                color="green",
                linewidth=2,
                linestyle="--"
            )
            ax.plot(
                self.data_transformed.index,
                self.data_decomposed[f"{column}__seasonal"],
                label="Сезонная компонента",
                color="black",
                linewidth=2,
                linestyle="--"
            )
            ax.plot(
                self.data_transformed.index,
                self.data_decomposed[f"{column}__resid"],
                label="Остаток",
                color="purple",
                linewidth=2,
                linestyle="--"
            )

            ax.set_title(f"{column}", fontsize=13)
            ax.set_ylabel("Значение", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=10)

        axes[-1].set_xlabel("Дата", fontsize=11)
        fig.suptitle("Разложение данных временного ряда", fontsize=15, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
