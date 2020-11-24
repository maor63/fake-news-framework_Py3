# Created by Aviad Elyashar at 23/04/2017

from preprocessing_tools.csv_importer import CsvImporter

class Kaggle_Importer_Parent(CsvImporter):
    def __init__(self, db):
        CsvImporter.__init__(self, db)
        self._author_type = str(self._config_parser.get(self.__class__.__name__, "author_type"))

