# Created by aviade      
# Time: 24/05/2016 11:27

import logging.config
from configuration.config_class import getConfig
import time
import csv
import os


###############################################################
# MODULES
###############################################################
moduleNames = {}
from DB.schema_definition import DB
moduleNames["DB"] = DB ## DB is special, it cannot be created using db.

from bad_actors_markup.bad_actors_markup import BadActorsMarkup
moduleNames["BadActorsMarkup"] = BadActorsMarkup

###############################################################
## SETUP
###############################################################
logging.config.fileConfig(getConfig().get("DEFAULT", "Logger_conf_file"))
domain = getConfig().get("DEFAULT", "domain")
logging.info("Start Execution ... ")
logging.info("SETUP global variables")

window_start = getConfig().eval("DEFAULT","start_date")
newbmrk = os.path.isfile("benchmark.csv")
bmrk_file = file("benchmark.csv","a")
bmrk_results = csv.DictWriter(bmrk_file,
                              ["time", "jobnumber", "config", "window_size", "window_start", "dones","posts",
                               "authors", "missing_required_authors", "actual_filled_authors"]  + list(moduleNames.keys()),
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
    logging.info("setup module: {0}".format(module))
    T = time.time()
    module.setUp()
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()

bmrk = {"config":getConfig().getfilename(), "window_start":"execute"}
for module in pipeline:
    logging.info("execute module: {0}".format(module))
    T = time.time()
    logging.info('*********Started executing '+module.__class__.__name__)

    module.execute(window_start)

    logging.info('*********Finished executing ' + module.__class__.__name__)
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()

if __name__ == '__main__':
    pass

