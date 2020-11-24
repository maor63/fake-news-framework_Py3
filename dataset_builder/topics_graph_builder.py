'''
@author: Jorge Bendahan jorgeaug@post.bgu.ac.il
'''
from preprocessing_tools.abstract_controller import AbstractController
from configuration.config_class import getConfig
from scipy import spatial
import time
import logging
from dataset_builder.graph_builder import GraphBuilder
import itertools

class GraphBuilder_Topic(GraphBuilder):
    """Generate graphs where nodes represent authors.
    There exists an edge between two authors if their topic vectors are close enough (measured by cosine similarity,
    'close enough' threshold is defined in config.ini"""

    def __init__(self, db):
        GraphBuilder.__init__(self, db)

        self._min_distance_threshold = float(self._config_parser.get(self.__class__.__name__, "min_distance_threshold"))
        self._distance_dict = {}

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting topics from DB ")

        if self._num_of_random_authors_for_graph is None:
            author_guid_topics_vector = self._db.get_authors_topics(self._domain, self._min_number_of_posts_per_author)
        else:
            #if not self._are_already_randomize_authors_for_graphs():
            #    self._db.randomize_authors_for_graph(self._min_number_of_posts_per_author, self._domain, self._num_of_random_authors_for_graph)

            author_guid_topics_vector = self._db.get_random_authors_topics(self._domain, self._min_number_of_posts_per_author)

        author_guids = list(author_guid_topics_vector.keys())

        possible_author_edges = list(itertools.combinations(author_guids, 2))
        for author1_guid, author2_guid in possible_author_edges:
            author1_guid_topic_vector = author_guid_topics_vector[author1_guid]
            author2_guid_topic_vector = author_guid_topics_vector[author2_guid]

            distance = spatial.distance.cosine(list(map(float, author1_guid_topic_vector)), list(map(float, author2_guid_topic_vector)))
            self.calculate_histogram_of_distances(distance)
            # print("distance: " + str(distance))
            # logging.info('*****distance between author: '+ author1[1]+' and author '+author2[1]+' is ' + str(distance)+'  ***** ')
            if distance <= float(self._min_distance_threshold):
                weight = 1 - distance
                self._create_and_optional_save_connection(author1_guid, author2_guid, weight)


        print("done computing similarities")
        print("start saving topic similarity edges in DB")
        self._db.save_author_connections(self._author_connections_edges)
        print("done saving topic similarity edges in DB")

    def calculate_histogram_of_distances(self, distance):
        #TODO: code review: WTF!!!!
        #replace code below with these two lines and test
        i = round(distance,1)
        self._distance_dict[i] = self._distance_dict.get(i,0)+1