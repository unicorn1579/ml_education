from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        time_column: str = 'date',
        target_frequency: str = None,
        fill_method: str = 'rolling',
        anomaly_method: str = 'rolling',
        rolling_window: int = 2,
        z_threshold: float = 3.0
    ):
        """
        time_column: str — имя столбца с временем
        target_frequency: str или None — желаемая частота ('H', 'D', 'T' и т.д.)
        fill_method: {'mean', 'ffill', 'rolling'} — метод заполнения пропусков
        anomaly_method: {'rolling', 'last'} — метод исправления выбросов
        rolling_window: int — окно для скользящего среднего
        z_threshold: float — порог z-score для выявления выбросов
        """
        self.time_column = time_column
        self.target_frequency = target_frequency
        self.fill_method = fill_method
        self.anomaly_method = anomaly_method
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.frequency = None
        self.values_columns = None


    def process(self, df):
        df_processed = df.copy()
        df_processed = self.prepare_datetime_column(df_processed)
        # Определяем частоту ряда
        self.frequency = pd.infer_freq(df_processed.index) or self.target_frequency
        # Изменяем гранулярность
        if self.target_frequency:
            df_processed = df_processed.resample(self.target_frequency).mean()
        # Отбор полей с числовыми значениями
        self.values_columns = df_processed.select_dtypes(include='number').columns
        df_processed = self.fill_missing(df_processed)
        df_processed = self.process_outliers(df_processed)
        return df_processed


    def prepare_datetime_column(self, df):
        """
        Преобразуем столбец времени в datetime
        """
        df[self.time_column] = pd.to_datetime(df[self.time_column], errors='coerce', utc=True)
        df = df.dropna(subset=[self.time_column])
        df = df.set_index(self.time_column).sort_index()
        return df

    def fill_missing(self, df):
        """
        Заполнение пропусков
        """
        if self.fill_method == 'mean':
            df[self.values_columns] = df[self.values_columns].fillna(df[self.values_columns].mean())
        elif self.fill_method == 'ffill':
            df[self.values_columns] = df[self.values_columns].ffill()
        elif self.fill_method == 'rolling':
            df[self.values_columns] = df[self.values_columns].fillna(
                df[self.values_columns].rolling(self.rolling_window, min_periods=1).mean()
            )
        return df

    def process_outliers(self, df):
        """
        Обработка выбросов
        """
        for col in self.values_columns:
            zscore = (df[col] - df[col].mean()) / df[col].std()
            mask = np.abs(zscore) > self.z_threshold
            if mask.any():
                if self.anomaly_method == 'rolling':
                    rolling_mean = df[col].rolling(self.rolling_window, min_periods=1).mean()
                    df.loc[mask, col] = rolling_mean[mask]
                elif self.anomaly_method == 'last':
                    df.loc[mask, col] = df[col].ffill()[mask]
        return df

    def fit(self, X=None, y=None):
        return self

    def transform(self, time_series: pd.DataFrame):
        if isinstance(time_series, pd.DataFrame):
            return self.process(time_series)
        elif isinstance(time_series, list):
            return [self.process(df) for df in time_series]
        else:
            raise ValueError("Input must be a DataFrame or a list of DataFrames.")

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)
