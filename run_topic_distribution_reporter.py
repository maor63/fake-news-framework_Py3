import logging
from configuration.config_class import getConfig
from topic_distribution_visualization.topic_distribution_visualization_generator import TopicDistrobutionVisualizationGenerator
from DB.schema_definition import *

if __name__ == '__main__':
    config_parser = getConfig()
    _db = DB()
    _db.setUp()

    topic_distribution_reporter = TopicDistrobutionVisualizationGenerator(_db)
    topic_distribution_reporter.create_topic_data_list()
    # topic_distribution_reporter.generate_visualization()
    print("Finished !!!!")
    logging.info("Finished !!!!")