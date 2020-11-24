#
# Created by Aviad on 16-April-17
#



import json

from DB.schema_definition import *
from commons.method_executor import Method_Executor
from preprocessing_tools.abstract_controller import AbstractController


class JSON_Importer_Parent(Method_Executor):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")
        self._json_path = self._config_parser.eval(self.__class__.__name__, "json_path")

    def _parse_json_files(self, currfolder):
        raw_json_file = open(self._json_path + currfolder)
        raw_json_str = raw_json_file.read()
        raw_json_data = json.loads(raw_json_str)
        return raw_json_data
