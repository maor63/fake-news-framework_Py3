
import logging
import time

from dataset_builder.graph_builders.bag_of_words_graph_builders.bag_of_words_graph_builder import \
    Bag_Of_Words_Graph_Builder

__author__ = "Aviad Elyashar"

class Bag_Of_Words_Graph_Builder_All_Combinations(Bag_Of_Words_Graph_Builder):
    """The edge between two authors is the jaccard similarity between the bag of words of each author"""

    def __init__(self, db):
        Bag_Of_Words_Graph_Builder.__init__(self, db)

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting posts from DB ")

        self.fill_author_guid_posts_dictionary()

        self.fill_author_guid_bag_of_words_dictionary_and_calculate_all_combinations()

        end_time = time.time()
        duration = end_time - start_time
        logging.info(" total time taken "+str(duration))

