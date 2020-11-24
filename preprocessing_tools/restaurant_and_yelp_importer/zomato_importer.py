#Written by Lior Bass 3/15/2018
import json

from configuration.config_class import getConfig


class Zomoto_Importer():
    def __init__(self, db):
        self._db = db
        self._config_parser = getConfig()
        self._files = self._config_parser.eval(self.__class__.__name__, "json_files")
        self._paths = self._config_parser.eval(self.__class__.__name__, "files_folder")
    def setUp(self):
        pass
    def is_well_defined(self):
        return True
    def execute(self, window_start=None):
        for file in self._files:
            file_path = self._paths+'\\'+file
            with open(file_path, 'r') as json_file:
                parsed_json = json.load(json_file)
                self._parse_file(parsed_json)
            pass

    def _parse_file(self, file):
        restaurants = file['restaurants']
        for row in file:
            self._parse_restaurant(row)

    def _parse_restaurant(self, row):
        pass

