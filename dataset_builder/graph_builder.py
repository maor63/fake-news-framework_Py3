# Created by aviade at 12/12/2016

from configuration.config_class import getConfig
from preprocessing_tools.abstract_controller import AbstractController
import time

class GraphBuilder(AbstractController):
    def __init__(self, db): #TODO: no kwargs in module constructors
        AbstractController.__init__(self, db)

        self._connection_type = str(self._config_parser.get(self.__class__.__name__, "connection_type"))
        self._max_objects_without_saving = self._config_parser.eval(self.__class__.__name__, "max_objects_without_saving")
        self._min_number_of_posts_per_author = self._config_parser.eval(self.__class__.__name__, "min_number_of_posts_per_author")
        self._num_of_random_authors_for_graph = self._config_parser.eval(self.__class__.__name__, "num_of_random_authors_for_graph")
        self._author_connections_edges = []
        # will save the connection and the reversed in order to make this connection type as un directed.
        self._existing_connections_dict = {}

    def setUp(self):
        if self._num_of_random_authors_for_graph is not None and not self._are_already_randomize_authors_for_graphs():
            self._db.randomize_authors_for_graph(self._min_number_of_posts_per_author, self._domain,
                                                 self._num_of_random_authors_for_graph)

        self._db.create_author_guid_num_of_posts_view()

    def execute(self, window_start):
        pass

    def _fill_author_connections(self, author_connections):
        #authors_guids_and_osn_ids = self._db.get_author_guid_and_author_osn_id(self._domain)
        #start_time = time.time()
        #print("execute started for " + self.__class__.__name__ + " started at " + str(start_time))

        for tuple in author_connections:
            guid_a = tuple[0]
            guid_b = tuple[1]
            weight = tuple[2]

            self._create_and_optional_save_connection(guid_a, guid_b, weight)



    def _create_and_optional_save_connection(self, author_guid_a, author_guid_b, weight):
        author_connection = self._db.create_author_connection(author_guid_a, author_guid_b, weight, self._connection_type,self._window_start)
        self._author_connections_edges.append(author_connection)
        if len(self._author_connections_edges) == self._max_objects_without_saving:
            self._db.save_author_connections(self._author_connections_edges)
            self._author_connections_edges = []

    def _are_already_randomize_authors_for_graphs(self):
        random_authors_for_graphs = self._db.get_random_authors_for_graphs()
        num_of_random_authors_for_graph = len(list(random_authors_for_graphs))

        return num_of_random_authors_for_graph != 0

    def _update_existing_connections_dictionary(self, author_guid, neighbor_author_guid):
        optional_key = author_guid + " - " + neighbor_author_guid
        self._existing_connections_dict[optional_key] = optional_key
        reversed_key = neighbor_author_guid + " - " + author_guid
        self._existing_connections_dict[reversed_key] = reversed_key

