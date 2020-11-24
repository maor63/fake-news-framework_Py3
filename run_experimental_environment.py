'''
Created on 06  SEP  2016

@author: Aviad Elyashar (aviade@post.bgu.ac.il)


This script is responsible for performing the following tasks:

'''
import logging.config
from configuration.config_class import getConfig
import time
import csv
import os


###############################################################
# MODULES
###############################################################
from experimental_environment.refactored_experimental_enviorment.classifier_trainer import Classifier_Trainer

moduleNames = {}
from DB.schema_definition import DB
moduleNames["DB"] = DB ## DB is special, it cannot be created using db.

from experimental_environment.experimental_environment import ExperimentalEnvironment
moduleNames["ExperimentalEnvironment"] = ExperimentalEnvironment


###############################################################
## SETUP
logging.config.fileConfig(getConfig().get("DEFAULT", "Logger_conf_file"))
#logger = logging.getLogger(getConfig().get("DEFAULT", "logger_name"))

logging.info("Start Execution ... ")

logging.info("SETUP global variables")

window_start = getConfig().eval("DEFAULT","start_date")
newbmrk = os.path.isfile("benchmark.csv")
bmrk_file = file("benchmark.csv","a")
bmrk_results = csv.DictWriter(bmrk_file,
                              ["time", "jobnumber", "config", "window_size", "window_start", "dones","retrieved_posts",
                               "pushed_key_posts"]  + list(moduleNames.keys()),
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

for module in pipeline:
    logging.info("execute module: {0}".format(module))
    T = time.time()
    logging.info('*********Started executing '+module.__class__.__name__)

    module.execute(window_start)
    classifier_trainer = Classifier_Trainer(db)
    classifier_trainer.execute()
    logging.info('*********Finished executing ' + module.__class__.__name__)
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

if __name__ == '__main__':
    pass
