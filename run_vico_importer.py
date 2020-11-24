#
# Created by Aviad on 03-Jun-16 11:29 AM.
#



import logging
from logging import config
from configuration.config_class import getConfig
import datetime
import time
import csv
import os
###############################################################
# MODULES
###############################################################
moduleNames = {}
from DB.schema_definition import DB
moduleNames["DB"] = DB ## DB is special, it cannot be created using db.

from preprocessing_tools.xml_importer import XMLImporter
moduleNames["XMLImporter"] = XMLImporter

from preprocessing_tools.create_authors_table import CreateAuthorTables
moduleNames["CreateAuthorTables"] = CreateAuthorTables

from preprocessing_tools.tumblr_importer.tumblr_importer import TumblrImporter
moduleNames["TumblrImporter"] = TumblrImporter


###############################################################
## SETUP
logging.config.fileConfig(getConfig().get("DEFAULT", "Logger_conf_file"))
logger = logging.getLogger(getConfig().get("DEFAULT", "logger_name"))

logger.info("Start Execution ... ")

logging.info("SETUP global variables")

window_start= getConfig().eval("DEFAULT","start_date")

end_date=getConfig().eval("DEFAULT","end_date")

window_size=datetime.timedelta(seconds=int(getConfig().get("DEFAULT","window_analyze_size_in_sec")))
step_size=datetime.timedelta(seconds=int(getConfig().get("DEFAULT","step_size_in_sec")))



newbmrk = os.path.isfile("benchmark.csv")
bmrk_file = file("benchmark.csv","a")
bmrk_results = csv.DictWriter(bmrk_file,
                              ["time", "jobnumber", "config", "window_size", "window_start", "dones","posts",
                               "authors"]  + list(moduleNames.keys()),
                              dialect="excel", lineterminator="\n")
if not newbmrk:
    bmrk_results.writeheader()


logging.info("CREATE pipeline")
db = DB()
moduleNames["DB"]=lambda x: x
pipeline=[]
for module in getConfig().sections():
    if moduleNames.get(module):
        pipeline.append(moduleNames.get(module)(db))


logging.info("SETUP pipeline")
bmrk = {"config":getConfig().getfilename(), "window_start":"setup"}



for module in pipeline:
    logger.info("setup module: {0}".format(module))
    T = time.time()
    module.setUp()
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()

for module in pipeline:
    logger.info("execute module: {0}".format(module))
    T = time.time()
    module.execute(window_start)
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

num_of_authors = db.get_number_of_authors()
bmrk["authors"] = num_of_authors

num_of_posts = db.get_number_of_posts()
bmrk["posts"] = num_of_posts

bmrk_results.writerow(bmrk)
bmrk_file.flush()

if __name__ == '__main__':
    pass

