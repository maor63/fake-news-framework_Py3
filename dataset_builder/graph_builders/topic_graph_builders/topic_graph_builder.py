from scipy import spatial
from dataset_builder.graph_builder import GraphBuilder
import itertools

__author__ = "Aviad Elyashar"

class Topic_Graph_Builder(GraphBuilder):

    def __init__(self, db):
        GraphBuilder.__init__(self, db)
        self._min_distance_threshold = self._config_parser.eval(self.__class__.__name__, "min_distance_threshold")
        self._author_guid_topics_vector_dict = {}

    def execute(self, window_start=None):
        pass

    def fill_author_guid_topics_vector_dictionary(self):
        self._author_guid_topics_vector_dict = self._db.get_authors_topics(self._domain,
                                                                           self._min_number_of_posts_per_author)

    def get_author_guids_from_dictionary(self):
        author_guids = list(self._author_guid_topics_vector_dict.keys())
        return author_guids

    def get_topic_vectors_from_dictionary(self):
        topic_vectors = list(self._author_guid_topics_vector_dict.values())
        return topic_vectors

    def calculate_cosine_distance_and_save_connection(self, author1_guid_topic_vector, author2_guid_topic_vector,
                                                      author1_guid, author2_guid):
        distance = spatial.distance.cosine(list(map(float, author1_guid_topic_vector)), list(map(float, author2_guid_topic_vector)))
        if distance <= float(self._min_distance_threshold):
            weight = 1 - distance
            self._create_and_optional_save_connection(author1_guid, author2_guid, weight)

    def calculate_all_combinations_and_save_topic_connections(self, author_guid_topics_vector):
        author_guids = list(author_guid_topics_vector.keys())

        possible_author_edges = list(itertools.combinations(author_guids, 2))
        for author1_guid, author2_guid in possible_author_edges:
            author1_guid_topic_vector = author_guid_topics_vector[author1_guid]
            author2_guid_topic_vector = author_guid_topics_vector[author2_guid]

            self.calculate_cosine_distance_and_save_connection(author1_guid_topic_vector, author2_guid_topic_vector,
                                                               author1_guid, author2_guid)

        print("done computing similarities")
        print("start saving topic similarity edges in DB")
        self._db.save_author_connections(self._author_connections_edges)
        print("done saving topic similarity edges in DB")