from DB.schema_definition import Post
from preprocessing_tools.abstract_controller import AbstractController
import os
import pandas as pd


class TableToCsvExporter(AbstractController):
    def __init__(self, db):
        super(TableToCsvExporter, self).__init__(db)
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._table_names = self._config_parser.eval(self.__class__.__name__, "table_names")

    def setUp(self):
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

    def execute(self, window_start=None):
        for table_name in self._table_names:
            self._export_table_data(table_name, os.path.join(self._output_path, table_name))

    def _export_table_data(self, db_table_name, export_csv_name):
        table = pd.read_sql_table(db_table_name, self._db.engine)
        table.to_csv(export_csv_name + '.csv', index=False)
