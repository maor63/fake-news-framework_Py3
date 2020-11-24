from dataset_builder.graph_builders.topic_graph_builders.topic_graph_builder import Topic_Graph_Builder
import time
import logging
import itertools

__author__ = "Aviad Elyashar"

class Topic_Graph_Builder_All_Combinations(Topic_Graph_Builder):
    """Generate graphs where nodes represent authors.
    There exists an edge between two authors if their topic vectors are close enough (measured by cosine similarity,
    'close enough' threshold is defined in config.ini"""

    def __init__(self, db):
        Topic_Graph_Builder.__init__(self, db)

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting topics from DB ")

        self.fill_author_guid_topics_vector_dictionary()
        author_guids = list(self._author_guid_topics_vector_dict.keys())

        possible_author_edges = list(itertools.combinations(author_guids, 2))
        for author1_guid, author2_guid in possible_author_edges:
            author1_guid_topic_vector = self._author_guid_topics_vector_dict[author1_guid]
            author2_guid_topic_vector = self._author_guid_topics_vector_dict[author2_guid]

            self.calculate_cosine_distance_and_save_connection(author1_guid_topic_vector, author2_guid_topic_vector,
                                                               author1_guid, author2_guid)


        print("done computing similarities")
        print("start saving topic similarity edges in DB")
        self._db.save_author_connections(self._author_connections_edges)
        print("done saving topic similarity edges in DB")
