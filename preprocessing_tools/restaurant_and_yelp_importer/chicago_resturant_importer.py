#Written by Lior Bass 2/18/2018
# coding=utf-8
from DB.schema_definition import Author
from preprocessing_tools.restaurant_and_yelp_importer.abstract_resturant_importer import Abstruct_Restaurant_Importer
from preprocessing_tools.restaurant_and_yelp_importer.yelp_importer import Yelp_Importer


class Chicago_Restaurant_Importer(Abstruct_Restaurant_Importer):
    def __init__(self, db):
        Abstruct_Restaurant_Importer.__init__(self, db)
        self._file_path = self._config_parser.eval(self.__class__.__name__, "file_path")
        self._yelp_importer = Yelp_Importer(db)

    def parse_row(self, row):
        try:
            author = Author()
            author.author_type = self.parse_type(row['Risk'])
            if author.author_type == -1:
                return None
            author.author_full_name = str(row['AKA Name']).encode('ascii', 'ignore').decode('ascii')
            author.location = str(row['City'] + ' ' + row['Address'])
            author.geo_enabled = str(row['Location'])
            author.name=row['DBA Name']
            author.created_at = str(row['Inspection Date'])
            return author
        except:
            return None

    def parse_type(self, type):
        if type == 'Risk 3 (Low)':
            return 0
        if type == 'Risk 2 (Medium)':
            return 1
        if type == 'Risk 1 (High)':
            return 2
        return -1

    def execute(self, window_start = None):
        self.abstract_parse_csv(',', self._file_path)