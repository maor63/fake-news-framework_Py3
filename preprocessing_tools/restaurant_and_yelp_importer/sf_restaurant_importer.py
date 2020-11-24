#Written by Lior Bass 2/18/2018
import logging

from DB.schema_definition import Author
from preprocessing_tools.restaurant_and_yelp_importer.abstract_resturant_importer import Abstruct_Restaurant_Importer


class SF_Restaurant_importer(Abstruct_Restaurant_Importer):
    def __init__(self, db):
        Abstruct_Restaurant_Importer.__init__(self, db)
        self._file_path = self._config_parser.eval(self.__class__.__name__, "file_path")

    def parse_row(self, row):
        try:
            author = Author()
            author.author_type = self.parse_type(row['risk_category'])
            if author.author_type == -1:
                return None
            author.author_full_name = str(row['business_name']).encode('ascii', 'ignore').decode('ascii')
            author.name = str(author.author_full_name).encode('ascii', 'ignore').decode('ascii')
            author.location = str(row['business_city'] + ' ' + row['business_address'])
            author.geo_enabled = str(row['business_location'])
            if author.geo_enabled is None or author.geo_enabled == "":
                return None
            author.created_at = str(row['inspection_date'])
            return author
        except:
            logging.info("error with row:"+ str(row))
            return None


    def parse_type(self, type):
        if type == 'Low Risk':
            return 0
        if type == 'Moderate Risk':
            return 1
        if type == 'High Risk':
            return 2
        return -1

    def execute(self, window_start = None):
        self.abstract_parse_csv(',', self._file_path)