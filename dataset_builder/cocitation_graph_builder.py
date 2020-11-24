'''
@author: Jorge Bendahan jorgeaug@post.bgu.ac.il
'''
import time
import logging
from dataset_builder.graph_builder import GraphBuilder


from commons.consts import Graph_Type

class GraphBuilder_CoCitation(GraphBuilder):
    """Generate graphs where nodes represent authors.
    There exists an edge between two authors if they reference the same post """

    def __init__(self, db):
        GraphBuilder.__init__(self, db)
        self._min_number_of_cocited_posts = int(self._config_parser.get(self.__class__.__name__,
                                                                    "min_number_of_cocited_posts"))

    def setUp(self):
        if self._num_of_random_authors_for_graph is not None and not self._are_already_randomize_authors_for_graphs():
            self._db.create_author_guid_num_of_posts_view()
            self._db.randomize_authors_for_graph(self._min_number_of_posts_per_author, self._domain,
                                                 self._num_of_random_authors_for_graph)
        self._db.create_author_post_cite_view()

    def execute(self, window_start):
        author_cocitations = self._db.get_cocitations(self._min_number_of_cocited_posts)
        if self._num_of_random_authors_for_graph is not None:
            random_author_citations = []
            random_author_guid_post_id_dict = self._db.get_random_author_guid_post_id_dictionary()
            for author_cocitation in author_cocitations:
                author1_guid = author_cocitation[0]
                author2_guid = author_cocitation[1]
                if author1_guid in random_author_guid_post_id_dict and author2_guid in random_author_guid_post_id_dict:
                    random_author_citations.append(author_cocitation)
            author_cocitations = random_author_citations

        if len(list(author_cocitations)) > 0:
            self._fill_author_connections(author_cocitations)

    def get_num_of_author_connections(self):
        return len(self._author_connections_edges)