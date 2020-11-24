# Created by aviade      
# Time: 22/06/2016 17:06

from configuration.config_class import getConfig
import logging
from DB.schema_definition import DB
from preprocessing_tools.abstract_controller import  AbstractController
from datetime import date, timedelta as td

class StatisticsProvider(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)

    def get_provided_vico_new_authors(self):
        min_date = self._db.get_post_min_date()
        max_date = self._db.get_post_max_date()


        delta = max_date - min_date

        total_numbers_of_new_users = []
        for i in range(delta.days + 1):
            current_date = min_date + td(days=i)
            new_author_screen_names = self._db.get_new_author_screen_names_by_date(min_date, current_date)
            number_of_new_users = len(new_author_screen_names)
            total_numbers_of_new_users.append(number_of_new_users)
            logging.info("Date: " + str(current_date) + " Amount of new users: " + str(len(new_author_screen_names)))
            logging.info(', '.join(new_author_screen_names))

        average_numbers_of_new_users = sum(total_numbers_of_new_users)/len(total_numbers_of_new_users)
        logging.info("The average number of new authors is: " + str(average_numbers_of_new_users))

        import numpy
        std = numpy.std(total_numbers_of_new_users, ddof=1)
        logging.info("The std of new authors is: " + str(std))



