
import logging
import time
from dataset_builder.graph_builders.bag_of_words_graph_builders.bag_of_words_graph_builder import \
    Bag_Of_Words_Graph_Builder
import copy
from scipy import spatial

__author__ = "Aviad Elyashar"

class Bag_Of_Words_Graph_Builder_K_Best(Bag_Of_Words_Graph_Builder):
    """The edge between two authors is the jaccard similarity between the bag of words of each author"""

    def __init__(self, db):
        Bag_Of_Words_Graph_Builder.__init__(self, db)
        self._k = self._config_parser.eval(self.__class__.__name__, "k")
        self._min_distance_threshold = self._config_parser.eval(self.__class__.__name__, "min_distance_threshold")
        # will save the connection and the reversed in order to make this connection type as un directed.

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting posts from DB ")

        self.fill_author_guid_posts_dictionary()
        self.fill_author_guid_bag_of_words_dictionary()
        self._fill_author_guid_bag_of_words_dict_dictionary()
        self._fill_author_guid_bag_of_words_vector_dictionary()

        author_guids = list(self._author_guid_bag_of_words_vector_dict.keys())
        bag_of_words_vectors = list(self._author_guid_bag_of_words_vector_dict.values())

        for author_guid in author_guids:
            current_bag_of_words_vectors = copy.deepcopy(bag_of_words_vectors)
            current_author_guids = copy.deepcopy(author_guids)

            author_guid_bag_of_words_vector = self._author_guid_bag_of_words_vector_dict[author_guid]
            author_guid_bag_of_words = self._author_guid_bag_of_words_dict[author_guid]

            index_to_remove = current_bag_of_words_vectors.index(author_guid_bag_of_words_vector)
            current_bag_of_words_vectors.pop(index_to_remove)
            current_author_guids.pop(index_to_remove)
            tree = spatial.KDTree(current_bag_of_words_vectors)
            result = tree.query(author_guid_bag_of_words_vector, k=self._k)
            nearest_neighbors_indexes = result[1]

            for nearest_neighbors_index in nearest_neighbors_indexes:
                neighbor_author_guid = current_author_guids[nearest_neighbors_index]

                optional_key = author_guid + " - " + neighbor_author_guid

                if optional_key not in self._existing_connections_dict:
                    neighbor_bag_of_words = self._author_guid_bag_of_words_dict[neighbor_author_guid]

                    self.calculate_jaccard_index_create_and_save_connection(author_guid, neighbor_author_guid,
                                                                            author_guid_bag_of_words,
                                                                            neighbor_bag_of_words)

                    self._update_existing_connections_dictionary(author_guid, neighbor_author_guid)
        self._db.save_author_connections(self._author_connections_edges)


        end_time = time.time()
        duration = end_time - start_time
        logging.info(" total time taken "+str(duration))

    def _fill_author_guid_bag_of_words_vector_dictionary(self):
        self._author_guid_bag_of_words_vector_dict = {}
        for author_guid, bag_of_word_dict in self._author_guid_bag_of_words_dict_dict.items():
            unique_words = list(self._word_dict.keys())
            self._author_guid_bag_of_words_vector_dict[author_guid] = []
            for word in unique_words:
                if word in bag_of_word_dict:
                    self._author_guid_bag_of_words_vector_dict[author_guid].append(1)
                else:
                    self._author_guid_bag_of_words_vector_dict[author_guid].append(0)

    def _fill_author_guid_bag_of_words_dict_dictionary(self):
        self._author_guid_bag_of_words_dict_dict = {}
        for author_guid, bag_of_words in  self._author_guid_bag_of_words_dict.items():
            self._author_guid_bag_of_words_dict_dict[author_guid] = {}
            for word in bag_of_words:
                if word not in self._author_guid_bag_of_words_dict_dict[author_guid]:
                    self._author_guid_bag_of_words_dict_dict[author_guid][word] = word

