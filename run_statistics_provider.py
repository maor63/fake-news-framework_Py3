# Created by aviade      
# Time: 22/06/2016 11:33

import logging
from configuration.config_class import getConfig
from DB.schema_definition import DB
from statistics_provider.statistics_provider import StatisticsProvider

if __name__ == '__main__':

    db = DB()
    db.setUp()
    statistics_provider = StatisticsProvider(db)

    statistics_provider.get_provided_vico_new_authors()


