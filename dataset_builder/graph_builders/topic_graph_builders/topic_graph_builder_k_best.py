from dataset_builder.graph_builders.topic_graph_builders.topic_graph_builder import Topic_Graph_Builder
from scipy import spatial
import time
import logging
from dataset_builder.graph_builder import GraphBuilder
import itertools
from scipy import spatial
import copy

__author__ = "Aviad Elyashar"

class Topic_Graph_Builder_K_Best(Topic_Graph_Builder):
    """Generate graphs where nodes represent authors.
    There exists an edge between two authors if their topic vectors are close enough (measured by cosine similarity,
    'close enough' threshold is defined in config.ini"""

    def __init__(self, db):
        Topic_Graph_Builder.__init__(self, db)
        self._k = self._config_parser.eval(self.__class__.__name__, "k")
        # will save the connection and the reversed in order to make this connection type as un directed.
        self._existing_connections_dict = {}

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting topics from DB ")

        self.fill_author_guid_topics_vector_dictionary()

        author_guids = self.get_author_guids_from_dictionary()
        topic_vectors = self.get_topic_vectors_from_dictionary()

        for author_guid in author_guids:
            current_topic_vectors = copy.deepcopy(topic_vectors)
            current_author_guids = copy.deepcopy(author_guids)

            author_guid_topic_vector = self._author_guid_topics_vector_dict[author_guid]
            index_to_remove = current_topic_vectors.index(author_guid_topic_vector)
            current_topic_vectors.pop(index_to_remove)
            current_author_guids.pop(index_to_remove)
            tree = spatial.KDTree(current_topic_vectors)
            result = tree.query(author_guid_topic_vector, k=self._k)
            nearest_neighbors_indexes = result[1]
            for nearest_neighbors_index in nearest_neighbors_indexes:
                neighbor_author_guid = current_author_guids[nearest_neighbors_index]

                optional_key = author_guid + " - " + neighbor_author_guid

                if optional_key not in self._existing_connections_dict:
                    neighbor_topic_vector = current_topic_vectors[nearest_neighbors_index]

                    self.calculate_cosine_distance_and_save_connection(author_guid_topic_vector,
                                                                       neighbor_topic_vector,
                                                                       author_guid, neighbor_author_guid)

                    self._update_existing_connections_dictionary(author_guid, neighbor_author_guid)


        print("done computing similarities")
        print("start saving topic similarity edges in DB")
        self._db.save_author_connections(self._author_connections_edges)
        print("done saving topic similarity edges in DB")

