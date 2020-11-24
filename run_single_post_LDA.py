import logging.config
import time

from configuration.config_class import getConfig

###############################################################
# MODULES
###############################################################
moduleNames = {}
from DB.schema_definition import DB
moduleNames["DB"] = DB ## DB is special, it cannot be created using db.

from dataset_builder.feature_extractor.topic_feature_generator import SinglePostLDA
moduleNames["SinglePostLDA"] = SinglePostLDA

###############################################################
## SETUP
###############################################################
logging.config.fileConfig(getConfig().get("DEFAULT", "Logger_conf_file"))
domain = getConfig().get("DEFAULT", "domain")
logging.info("Start Execution ... ")
logging.info("SETUP global variables")

window_start = getConfig().eval("DEFAULT","start_date")


logging.info("CREATE pipeline")
db = DB()
moduleNames["DB"]=lambda x: x
pipeline=[]
for module in getConfig().sections():
    if moduleNames.get(module):
        pipeline.append(moduleNames.get(module)(db))

logging.info("SETUP pipeline")


for module in pipeline:
    logging.info("setup module: {0}".format(module))
    T = time.time()
    module.setUp()
    T = time.time() - T


for module in pipeline:
    logging.info("execute module: {0}".format(module))
    T = time.time()
    logging.info('*********Started executing '+module.__class__.__name__)

    module.execute(window_start)

    logging.info('*********Finished executing ' + module.__class__.__name__)
    T = time.time() - T

if __name__ == '__main__':
    pass

