from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd


class TableDuplicationRemover(AbstractController):
    def __init__(self, db):
        super(TableDuplicationRemover, self).__init__(db)
        self._table_name = self._config_parser.eval(self.__class__.__name__, "table_name")
        self._table_ids = self._config_parser.eval(self.__class__.__name__, "table_keys")

    def execute(self, window_start):
        table = pd.read_sql_table(self._table_name, self._db.engine)
        table.drop_duplicates(subset=self._table_ids, keep='first', inplace=True)
        table.to_sql(self._table_name, con=self._db.engine, if_exists='replace', index=False)

