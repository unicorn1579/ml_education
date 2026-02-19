from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb


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
class ModelMetadata:
    """
    Инициализирует объект ModelMetadata.

    Args:
        model_type (str): тип модели
        model (LinearRegression | XGBRegressor): модель
        y_test (): тестовый набор данных
        y_prediction (): спрогнозированный набор данных
        metrics (Dict[str, float]): метрики качества модели
    """
    model_type: str = None
    model: LinearRegression = None
    x: pd.DataFrame = None
    y: pd.Series = None  
    x_test: pd.DataFrame = None
    y_test: pd.Series = None
    y_prediction: pd.Series = None
    metrics: Dict[str, float] = None
    time_column: str = None

    def predict(self, x_data: pd.DataFrame, y_data: pd.Series, metric_name: str = 'FA_MAPE') -> None:
        """
        Прогноз
        """
        y_prediction = self.model.predict(x_data)
        metrics={
            "MAE": mean_absolute_error(y_data, y_prediction),
            "MAPE": mean_absolute_percentage_error(y_data, y_prediction),
            "WAPE": np.sum(np.abs(y_data - y_prediction)) / np.sum(y_data),
            "FA_MAPE": 1 - mean_absolute_percentage_error(y_data, y_prediction),
            "FA_WAPE": 1 - np.sum(np.abs(y_data - y_prediction)) / np.sum(y_data)
        }
                        
        if metric_name is None:
            test_metrics_str = ", ".join(
                f"{name}={value:.5f}" for name, value in self.metrics.items()
            )
            valid_metrics_str = ", ".join(
                f"{name}={value:.5f}" for name, value in metrics.items()
            )
        else:
            test_metrics_str = f"{metric_name}={self.metrics[metric_name]:.5f}"
            valid_metrics_str = f"{metric_name}={metrics[metric_name]:.5f}"

        plt.plot(y_data.index,y_data, color="black", alpha=0.5, label="Observed")
        plt.plot(
            y_data.index,
            y_prediction,
            color="red",
            label=f"Predicted (test_{test_metrics_str}, valid_{valid_metrics_str})"
        )

        plt.title(self.model_type)
        plt.grid(True)
        plt.legend()

        if self.time_column:
            plt.xlabel(self.time_column)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def show(self, metric_name: str = 'FA_WAPE') -> None:
        """
        Визуализирует модель
        """
        if metric_name is None:
            metrics_str = ", ".join(
                f"{name}={value:.5f}" for name, value in self.metrics.items()
            )
        else:
            metrics_str = f"{metric_name}={self.metrics[metric_name]:.5f}"

        plt.plot(self.y_test.index, self.y_test, color="black", alpha=0.5, label="Observed")
        plt.plot(
            self.y_test.index,
            self.y_prediction,
            color="red",
            label=f"Predicted ({metrics_str})"
        )

        plt.title(self.model_type)
        plt.grid(True)
        plt.legend()

        if self.time_column:
            plt.xlabel(self.time_column)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

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
    data_processed_columns: Dict[str, pd.DataFrame] = None

    correlation_method: str = "spearman"
    correlation_threshold: float = 0.3

    models_metadata: Dict[str, List[ModelMetadata]] = None

    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    eval_metric: str = "mae"
    early_stopping_rounds: int = 20
    
    train_ratio: float = 0.9
    cv_method: str = "expanding"
    cv_freq: str = "Q"

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
        self.features_generation()
        self.select_significant_features()

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
        self.data_processed_columns = {}
        for column in self.processed_columns:
            self.data_processed_columns[column] = self.data_transformed[[column]].copy()
            column_decomposed = seasonal_decompose(
                self.data_transformed[column],model=self.decompose_model,period=self.decompose_period
            )
            data_decomposed[f"{column}__observed"] = column_decomposed.observed
            data_decomposed[f"{column}__trend"] = column_decomposed.trend
            data_decomposed[f"{column}__seasonal"] = column_decomposed.seasonal
            data_decomposed[f"{column}__resid"] = column_decomposed.resid

            self.data_processed_columns[column]["observed"] = column_decomposed.observed
            self.data_processed_columns[column]["trend"] = column_decomposed.trend
            self.data_processed_columns[column]["seasonal"] = column_decomposed.seasonal
            self.data_processed_columns[column]["resid"] = column_decomposed.resid
            self.data_processed_columns[column] = self.data_processed_columns[column].drop(column, axis=1)
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
        fig.suptitle(f"Разложение данных временного ряда {self.file_name}", fontsize=15, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.xticks(rotation=45)
        plt.show()

    def features_generation(self) -> None:
        """
        Генерация признаков применительно к каждой обрабатываемой колонки исходных данных
        """
        for column in self.processed_columns:
            # Календарные признаки
            self.data_processed_columns[column]["day"] = self.data_processed_columns[column].index.day
            self.data_processed_columns[column]["dayofweek"] = self.data_processed_columns[column].index.dayofweek  # 0-6
            self.data_processed_columns[column]["weekofyear"] = self.data_processed_columns[column].index.isocalendar().week.astype(int)
            self.data_processed_columns[column]["month"] = self.data_processed_columns[column].index.month
            self.data_processed_columns[column]["quarter"] = self.data_processed_columns[column].index.quarter
            # Циклические признаки
            self.data_processed_columns[column]["dow_sin"] = np.sin(2 * np.pi * self.data_processed_columns[column].index.dayofweek / 7)
            self.data_processed_columns[column]["dow_cos"] = np.cos(2 * np.pi * self.data_processed_columns[column].index.dayofweek / 7)
            self.data_processed_columns[column]["month_sin"] = np.sin(2 * np.pi * self.data_processed_columns[column].index.month / 12)
            self.data_processed_columns[column]["month_cos"] = np.cos(2 * np.pi * self.data_processed_columns[column].index.month / 12)
            self.data_processed_columns[column]["dayofyear_sin"] = np.sin(2 * np.pi * self.data_processed_columns[column].index.dayofyear / 365)
            self.data_processed_columns[column]["dayofyear_cos"] = np.cos(2 * np.pi * self.data_processed_columns[column].index.dayofyear / 365)
            self.data_processed_columns[column]["hour_sin"] = np.sin(2 * np.pi * self.data_processed_columns[column].index.hour / 24)
            self.data_processed_columns[column]["hour_cos"] = np.cos(2 * np.pi * self.data_processed_columns[column].index.hour / 24)
            # Признаки сезонности
            # сила сезонного эффекта
            self.data_processed_columns[column]["seasonal__abs"] = self.data_processed_columns[column]["seasonal"].abs()
            # фаза сезонности
            self.data_processed_columns[column]["seasonal__phase"] = self.data_processed_columns[column]["seasonal"].rank(pct=True)
            # дельта сезонности
            self.data_processed_columns[column]["seasonal__diff"] = self.data_processed_columns[column]["seasonal"].diff()
            # нормализованная сезонность
            self.data_processed_columns[column]["seasonal__scaled"] = (self.data_processed_columns[column]["seasonal"] - self.data_processed_columns[column]["seasonal"].mean()) / self.data_processed_columns[column]["seasonal"].std()

    def select_significant_features(self) -> None:
        for column in self.processed_columns:
            # Нормализация только тренда
            trend = self.data_processed_columns[column]["trend"]
            self.data_processed_columns[column]["trend"] = (trend - trend.mean()) / trend.std(ddof=0)

            # Корреляция признаков с данными наблюдений (observed)
            correlation = self.data_processed_columns[column].corr(method=self.correlation_method)[["observed"]].drop("observed")
            # Отбор значимых признаков
            significant = correlation[correlation["observed"].abs() > self.correlation_threshold]
            selected_features = significant.index.tolist() + ["observed"]
            self.data_processed_columns[column] = self.data_processed_columns[column][selected_features]

    def generate_splits(self, data: pd.DataFrame):
        periods = data.index.to_period(self.cv_freq)
        unique_periods = periods.unique()

        for i in range(1, len(unique_periods)):

            if self.cv_method == "expanding":
                train_mask = periods < unique_periods[i]
                test_mask = periods == unique_periods[i]

            elif self.cv_method == "sliding":
                train_mask = periods == unique_periods[i - 1]
                test_mask = periods == unique_periods[i]

            else:
                raise ValueError(f"Unknown CV method: {self.cv_method}")

            yield data.loc[train_mask], data.loc[test_mask]

    def rolling_models(self) -> None:
        self.models_metadata = dict()
        
        for column in self.processed_columns:
            self.models_metadata[column] = list()

            data = self.data_processed_columns[column].dropna()
            x = data.drop(columns=["observed"])
            y = data["observed"]

            for train, test in self.generate_splits(data):

                # разделение на признаки и таргет
                x_train = train.drop(columns=["observed"])
                y_train = train["observed"]
                x_test = test.drop(columns=["observed"])
                y_test = test["observed"]

                # LinearRegression
                #model = LinearRegression()
                #model.fit(x_train, y_train)
                #
                #y_prediction = model.predict(x_test)
                #
                #self.models_metadata[column].append(
                #    ModelMetadata(
                #        model_type='LinearRegression',
                #        model=model,
                #        time_column=self.time_column,
                #        x_test=x_test,
                #        y_test=y_test,
                #        x=x,
                #        y=y,
                #        y_prediction=y_prediction,
                #        metrics={
                #            "MAE": mean_absolute_error(y_test, y_prediction),
                #            "MAPE": mean_absolute_percentage_error(y_test, y_prediction),
                #            "WAPE": np.sum(np.abs(y_test - y_prediction)) / np.sum(y_test),
                #            "FA_MAPE": 1 - mean_absolute_percentage_error(y_test, y_prediction),
                #            "FA_WAPE": 1 - np.sum(np.abs(y_test - y_prediction)) / np.sum(y_test)
                #        }
                #    )
                #)

                # XGBRegressor
                split_idx = int(len(x_train) * 0.8)
                x_tr = x_train.iloc[:split_idx]
                y_tr = y_train.iloc[:split_idx]
                x_val = x_train.iloc[split_idx:]
                y_val = y_train.iloc[split_idx:]
                
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    eval_metric=self.eval_metric,
                    early_stopping_rounds=self.early_stopping_rounds
                )
                model
                model.fit(
                    x_tr, y_tr,
                    eval_set=[(x_val, y_val)],
                    verbose=False
                )

                y_prediction = model.predict(x_test)

                self.models_metadata[column].append(
                    ModelMetadata(
                        model_type='XGBRegressor',
                        model=model,
                        time_column=self.time_column,
                        x_test=x_test,
                        y_test=y_test,
                        x=x,
                        y=y,
                        y_prediction=y_prediction,
                        metrics={
                            "MAE": mean_absolute_error(y_test, y_prediction),
                            "MAPE": mean_absolute_percentage_error(y_test, y_prediction),
                            "WAPE": np.sum(np.abs(y_test - y_prediction)) / np.sum(y_test),
                            "FA_MAPE": 1 - mean_absolute_percentage_error(y_test, y_prediction),
                            "FA_WAPE": 1 - np.sum(np.abs(y_test - y_prediction)) / np.sum(y_test)
                        }
                    )
                )

    def show_metric(self, column_name: str) -> None:
        n_models = len(self.models_metadata[column_name])
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 3 * n_models), sharex=True)
        results = [(model_metadata.model_type, model_metadata.y_test, model_metadata.y_prediction, model_metadata.metrics)
                   for model_metadata in self.models_metadata[column_name]]

        for ax, (model_type, y_test, y_prediction, metrics) in zip(axes, results):
            metrics_str = ",\n".join(f"{name}={value:.5f}" for name, value in metrics.items())
            ax.plot(y_test.index, y_test, color="black", alpha=0.5, label="Observed")
            ax.plot(y_test.index, y_prediction, color="red", label=f"Predicted ({metrics_str})")
            ax.set_title(model_type)
            ax.grid(True)
            ax.legend()

        plt.xlabel(self.time_column)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

    def get_effective_model(self, column_name: str, metric_name: str = 'MAE') -> ModelMetadata:
        effective_model = None

        for model_metadata in self.models_metadata[column_name]:
            if not effective_model:
                effective_model = model_metadata
            else:
                if metric_name == 'MAE' and model_metadata.metrics[metric_name] < effective_model.metrics[metric_name]:
                    effective_model = model_metadata

        return effective_model

