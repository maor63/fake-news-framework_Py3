
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.abstract_controller import AbstractController
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis



class ClaimGraphFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._graph_types = self._config_parser.eval(self.__class__.__name__, "graph_types")
        self._node_algorithms = self._config_parser.eval(self.__class__.__name__, "node_algorithms")
        self._directed_node_algorithms = self._config_parser.eval(self.__class__.__name__, "directed_node_algorithms")
        self._aggregations = self._config_parser.eval(self.__class__.__name__, "aggregations")
        self._graph_algorithms = self._config_parser.eval(self.__class__.__name__, "graph_algorithms")
        self._prefix = self.__class__.__name__

    def execute(self, window_start=None):
        claim_post_author_connection = self._db.get_claim_post_author_connections()
        claim_author_guid_dict = defaultdict(list)
        for claim_id, post, author_guid in claim_post_author_connection:
            claim_author_guid_dict[claim_id].append(author_guid)

        for i, graph_type in enumerate(self._graph_types):
            print('Graph type: {}, {}/{}'.format(graph_type, str(i + 1), len(self._graph_types)))
            type_connections = self._db.get_author_connections_by_type(graph_type)
            type_edges = [(ac[0], ac[1]) for ac in type_connections]
            connection_type_graph = nx.Graph()
            connection_type_graph.add_edges_from(type_edges)

            # plt.figure(figsize=(20, 14))
            # pos = nx.fruchterman_reingold_layout(connection_type_graph)
            # nx.draw(connection_type_graph, pos=pos)
            # plt.show()

            # nx.write_gexf(connection_type_graph, '{}.gexf'.format(graph_type))
            # continue
            print('Calculate for Undirected graph')
            self.compute_graph_features(claim_author_guid_dict, connection_type_graph, self._node_algorithms,
                                        '_undirected')
            print('Calculate for Directed graph')
            self.compute_graph_features(claim_author_guid_dict, nx.DiGraph(connection_type_graph),
                                        self._directed_node_algorithms, '_directed')

    def compute_graph_features(self, claim_author_guid_dict, connection_type_graph, node_algorithms, suffix):
        author_features = []
        for j, (claim_id, author_guids) in enumerate(claim_author_guid_dict.items()):
            print('\rgenerate feature for claim, {}/{}'.format(str(j + 1), len(claim_author_guid_dict)), end='')
            claim_graph = connection_type_graph.subgraph(author_guids)
            for node_algorithm in node_algorithms:
                nodes_result = getattr(nx, node_algorithm)(claim_graph)
                for aggregation in self._aggregations:
                    value = eval(aggregation)(list(nodes_result.values()))
                    feature_name = '{}_{}'.format(node_algorithm, aggregation)
                    author_features.append(self._create_feature(feature_name, claim_id, value, suffix))
                if len(author_features) > 1000:
                    self._db.add_author_features_fast(author_features)
                    author_features = []
        print()
        self._db.add_author_features_fast(author_features)
