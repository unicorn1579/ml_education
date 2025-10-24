import json
from pathlib import Path
from typing import List

import pandas as pd

from modules.common import TimeSeries, InputParams


class Reader:
    def __init__(self, params: InputParams) -> None:
        self.params: InputParams = params
        self.full_files_names = self.get_full_files_names()

        self.time_series: TimeSeries | None = TimeSeries()

    def get_full_files_names(self) -> List[str]:
        if self.params.tickets_group:
            with open('configuration.json', 'r') as file:
                configuration = json.load(file)
            files_names = configuration['tickets_groups'][self.params.tickets_group]
        else:
            files_names = self.params.files_names

        return self.add_extension(files_names)

    @staticmethod
    def add_extension(files_names: List[str]) -> List[str]:
        return [f'{file}.csv' for file in files_names]

    def run(self) -> None:
        self.get_data_from_files()

    def get_data_from_files(self):
        self.time_series.directory = self.params.directory
        self.time_series.files_names = self.full_files_names
        for file_name in self.full_files_names:
            file_path = Path(self.params.directory) / file_name
            df = pd.read_csv(file_path)
            df['source_file'] = file_name
            self.time_series.data.append(df)
