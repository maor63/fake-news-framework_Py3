#Written by Lior Bass 2/18/2018

import calendar
import codecs
import csv
import logging


# from datetime import datetime
from datetime import datetime

import time

from DB.schema_definition import Author
from configuration.config_class import getConfig
from preprocessing_tools.restaurant_and_yelp_importer.yelpapi import YelpAPI


class Abstruct_Restaurant_Importer():
    def __init__(self, db):
        self._db = db
        self._config_parser = getConfig()
        self._commit_threshold = 500

        self._yelp_api = YelpAPI()
        self._radius = 100
        self._print_threshold = 30

    def abstract_parse_csv(self, delimiter, file_path):
        logging.info("open csv with encoding='ascii', mode='r', errors='ignore' ")
        f = codecs.open(file_path, encoding='ascii', mode='r', errors='ignore')
        logging.info("IMPORT CSV %s"%f)
        try:
            reader = csv.DictReader(f,delimiter=delimiter)
            logging.info("opened file")
            rest_dict = {}
            for row in reader:
                try:
                    self._row = row
                    self._print_info()
                    author = self.parse_row(row)
                    if author is None:
                        self._skip_row()
                        # logging.info("author is none")
                        continue
                    lat, int = self.extact_coardinates(author)
                    if lat is None or int is None:
                        self._skip_row()
                        # logging.info("failed to get location")
                        continue
                    yelp_id = self._yelp_api.get_restaurants_id(author.name, int, lat,self._radius)
                    if yelp_id is None:
                        self._skip_row()
                        # logging.info("failed to get yelp id")
                        continue
                    author.author_guid = yelp_id
                    if author.author_guid in rest_dict:
                        self._skip_row()
                        # logging.info("author guid in rest dict")
                        continue
                    author.domain = 'Restaurant'
                    rest_dict[author.author_guid]= author
                    self._valid_counter +=1
                    if self._valid_counter % self._commit_threshold==0:
                        logging.info("valid restaurants processed: "+str(self._valid_counter))
                        self._db.add_authors(list(rest_dict.values()))
                except Exception as exc:
                    self._skip_row()
                    logging.info(exc)
                    logging.info("encounter unknown error")
            self._db.add_authors(list(rest_dict.values()))
            self._db.commit()

        except Exception as e:
            logging.exception(e)
            logging.exception("error with row: "+str(self._row))
        finally:
            f.close()

    def extact_coardinates(self, coordinates_text):
        try:
            cor = coordinates_text.geo_enabled.replace('(', "")
            cor = cor.replace(')', "")
            cor = cor.split(', ')
            lat = float(cor[0])
            long = float(cor[1])
            if lat==0 or int==0:
                return None, None
            return lat, int
        except:
            return None, None

    def _skip_row(self):
        self._skipped_rows+=1

    def _print_info(self):
        try :
            self._row_counter += 1
            new_time = calendar.timegm(time.gmtime())
            if new_time - self._old_time > self._print_threshold:
                logging.info(
                    " processed rows: " + str(self._row_counter-1) + " valid rows: " + str(self._valid_counter) + " skipped_rows: " + str(
                        self._skipped_rows))
                self._old_time = new_time
                return
        except:
            self._row_counter = 1
            self._valid_counter = 0
            self._skipped_rows = 0
            self._old_time = calendar.timegm(time.gmtime())
            logging.info("initialized counters")
            return

    def setUp(self):
        pass
    def is_well_defined(self):
        return True