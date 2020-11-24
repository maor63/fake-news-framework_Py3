from scipy import spatial
import time
import logging
import itertools

from dataset_builder.graph_builders.topic_graph_builders.topic_graph_builder import Topic_Graph_Builder

__author__ = "Aviad Elyashar"

class Topic_Graph_Builder_Random_Combinations(Topic_Graph_Builder):
    def __init__(self, db):
        Topic_Graph_Builder.__init__(self, db)

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting topics from DB ")

        author_guid_topics_vector = self._db.get_random_authors_topics(self._domain, self._min_number_of_posts_per_author)

        self.calculate_all_combinations_and_save_topic_connections(author_guid_topics_vector)

