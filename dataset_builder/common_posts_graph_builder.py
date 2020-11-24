'''
@author: Aviad Elyashar aviade@post.bgu.ac.il
'''

from configuration.config_class import getConfig
import time
import logging
from dataset_builder.graph_builder import GraphBuilder
import re
import networkx as nx
from networkx.algorithms import bipartite
from DB.schema_definition import AuthorConnection
from commons.commons import *


from commons.consts import Graph_Type, Domains

class GraphBuilder_Common_Posts(GraphBuilder):
    """Generate graphs where nodes represent authors.
    There exists an edge between two authors if they reference the same post """

    def __init__(self, db):
        GraphBuilder.__init__(self, db)

        self._min_number_of_common_posts = int(self._config_parser.get(self.__class__.__name__,
                                                                    "min_number_of_common_posts"))
        self._min_number_of_posts_per_author = int(self._config_parser.get(self.__class__.__name__,
                                                                    "min_number_of_posts_per_author"))


    def execute(self, window_start):
        author_timelines_tuples = self._db.get_author_timelines_by_min_num_of_posts(self._domain, self._min_number_of_posts_per_author)
        print('retrieving authors and posts')
        if self._num_of_random_authors_for_graph is not None:
            random_author_timelines_tuples = []
            random_author_guid_post_id_dict = self._db.get_random_author_guid_post_id_dictionary()
            for author_timeline in author_timelines_tuples:
                author_guid = author_timeline[0]
                if author_guid in random_author_guid_post_id_dict:
                    random_author_timelines_tuples.append(author_timeline)

            author_timelines_tuples = random_author_timelines_tuples

        author_post_tuples, author_type_dict = self._create_author_post_and_author_type_tuples(author_timelines_tuples)

        print('creating bi-partite graph')
        bi_graph = self._create_bi_graph(author_post_tuples)
        author_guids, posts = list(zip(*author_post_tuples))
        del author_post_tuples
        author_guids = list(set(author_guids))

        print('creating authors graph')
        authors_projection_graph = bipartite.projected_graph(bi_graph, author_guids, multigraph=True)
        del bi_graph

        print('counting shared posts')

        source_author_guid_dest_author_guid_shared_posts_dict = self._count_shared_posts(authors_projection_graph)

        self._fill_common_posts_connections(source_author_guid_dest_author_guid_shared_posts_dict)


    def _create_author_post_and_author_type_tuples(self, author_timelines_tuples):
        author_post_tuples = []
        author_type_dict = {}

        for tuple in author_timelines_tuples:
            raw_content = tuple[1]
            content = self._clean_content(raw_content)
            if content is not None:
                author_guid = tuple[0]
                author_type = tuple[2]
                author_post_tuples.append([author_guid, content])
                author_type_dict[author_guid] = author_type

        return author_post_tuples, author_type_dict

    def _clean_content(self, post_content):
        if post_content is not None:
            content = re.sub(r'https?:.*', '', post_content, flags=re.MULTILINE)
            content = re.sub(r'<em>', '', content, flags=re.MULTILINE)
            content = re.sub(r'</em>', '', content, flags=re.MULTILINE)
            if len(content) > 10:
                content = str(content.encode('utf-8'))
                return content
            else:
                return None
        else:
            return None

    def _create_bi_graph(self, post_author_tuples):
        bi_graph = nx.Graph()
        bi_graph.add_edges_from(post_author_tuples)

        return bi_graph

    def _count_shared_posts(self, authors_projection_graph):
        edge_list = authors_projection_graph.edges()
        source_author_guid_dest_author_name_shared_posts_dict = {}

        for edges in edge_list:
            source_author_guid = edges[0]
            destination_author_guid = edges[1]

            self._update_dictionary(source_author_guid, destination_author_guid, source_author_guid_dest_author_name_shared_posts_dict)
            #self._update_dictionary(destination_author_name, source_author_name, source_author_name_dest_author_name_shared_posts_dict)

        return source_author_guid_dest_author_name_shared_posts_dict

    def _update_dictionary(self, source_author_guid, destination_author_guid,
                           source_author_guid_dest_author_guid_shared_posts_dict):
        if source_author_guid not in source_author_guid_dest_author_guid_shared_posts_dict:
            source_author_guid_dest_author_guid_shared_posts_dict[source_author_guid] = {}
            source_author_guid_dest_author_guid_shared_posts_dict[source_author_guid][destination_author_guid] = 1

        elif destination_author_guid not in source_author_guid_dest_author_guid_shared_posts_dict[source_author_guid]:
            source_author_guid_dest_author_guid_shared_posts_dict[source_author_guid][destination_author_guid] = 1
        else:
            source_author_guid_dest_author_guid_shared_posts_dict[source_author_guid][destination_author_guid] += 1

    def _fill_common_posts_connections(self, source_author_osn_id_dest_author_osn_id_shared_posts_dict):
        author_connections = []
        count = 0
        for source_author_osn_id, dest_author_osn_id_shared_posts_dict in source_author_osn_id_dest_author_osn_id_shared_posts_dict.items():
            for dest_author_osn_id, shared_posts in dest_author_osn_id_shared_posts_dict.items():
                count += 1
                weight = float(shared_posts)
                self._create_and_optional_save_connection(source_author_osn_id, dest_author_osn_id, weight)

        self._db.save_author_connections(self._author_connections_edges)
        print(("Number of common posts connections: " + str(len(author_connections))))
