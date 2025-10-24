from dataclasses import dataclass, field
from typing import List, NamedTuple

import pandas as pd


class InputParams(NamedTuple):
    """
    Входные параметры
    """
    directory: str
    files_names: str
    tickets_group: str
    time_column: str
    target_frequency: str
    fill_method: str
    anomaly_method: str
    rolling_window: int


@dataclass
class TimeSeries:
    """
    Данные временных рядов
    """
    directory: str = None
    files_names: List[str] = field(default_factory=lambda: list())
    data: List[pd.DataFrame] = field(default_factory=lambda: list())
    data_transformed: List[pd.DataFrame] = field(default_factory=lambda: list())

    def show(self) -> None:
        """
        Демонстрация данных
        """
        print(self.directory)
        print(self.files_names)


