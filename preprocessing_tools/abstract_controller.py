# Created by aviade      
# Time: 02/05/2016 14:59

import abc
from abc import ABCMeta, abstractmethod
from configuration.config_class import getConfig

import logging
from logging import config
import datetime


# import time
# from commons import date_to_str,str_to_date

class AbstractController(object):
    def __init__(self, db):
        self._config_parser = getConfig()
        self._db = db

        logging.config.fileConfig(self._config_parser.get("DEFAULT", "logger_conf_file"))
        self.logger = logging.getLogger(self._config_parser.get("DEFAULT", "logger_name"))
        self._start_date = self._config_parser.get(self.__class__.__name__, "start_date")
        self._window_start_query = self._config_parser.get(self.__class__.__name__, "start_date").strip("date('')")
        self._window_size = datetime.timedelta(seconds=int(self._config_parser.get(self.__class__.__name__, "window_analyze_size_in_sec")))
        self.keep_results_for = datetime.timedelta(seconds=self._config_parser.getint(self.__class__.__name__, "keep_results_for"))
        self._window_start = self._config_parser.eval(self.__class__.__name__, "start_date")
        self._social_network_url = self._config_parser.eval("DEFAULT", 'social_network_url')
        self._domain = str(self._config_parser.get(self.__class__.__name__, "domain"))
        # self._window_start_query = str_to_date(date_to_str(self._start_date,formate=self.DB_date_format),formate=self.DB_date_format)

    @property
    def _window_end(self):
        return self._window_start + self._window_size

    #     @property
    #     def _window_end_query(self):
    #         return str_to_date(date_to_str(self._window_end,formate=self.DB_date_format),formate=self.DB_date_format)
    #



    def setUp(self):
        pass

    def execute(self, window_start):
        self._window_start = window_start

    def cleanUp(self, window_start):
        pass

    def canProceedNext(self, window_start):
        return True

    def tearDown(self):
        pass

    def is_well_defined(self):
        return True

    def check_config_has_attributes(self, attr_list):
        for attribute in attr_list:
            attr_in_config = self._config_parser.get(self.__class__.__name__, attribute)
            if attr_in_config is None or len(attr_in_config) > 0:
                raise Exception("missing expected parameter in config: "+attribute)
        return True