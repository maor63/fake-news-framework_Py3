
import numpy as np
import logging
import sys
import networkx as nx
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from sklearn.utils.multiclass import unique_labels


class FlickrFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **{'authors': [], 'posts': {}})
        self._features = self._config_parser.eval(self.__class__.__name__, "feature_list")

    def execute(self, window_start=None):
        authors_features = []
        graphs = self.load_graphs()
        for action_name in self._features:
            authors_features += getattr(self, action_name)(graphs)
        self._db.add_author_features_fast(authors_features)

    def load_graphs(self):
        posts = self._db.get_posts_filtered_by_domain('flickr')
        author_connections_dict = self._db.get_author_connections_dict()
        authors_by_domain_dict = self._db.get_authors_by_domain_dict()
        graphs = []
        for i, post in enumerate(posts):
            print('\r create graph {}/{}'.format(str(i + 1), len(posts)), end='')
            if post.post_type is not None:
                graph_id = post.post_osn_id
                connections = author_connections_dict.get(str(graph_id), [])
                edges = [(connection.source_author_guid, connection.destination_author_guid) for connection in
                         connections]
                nodes = [author.author_guid for author in authors_by_domain_dict.get(str(graph_id), [])]
                graph = nx.DiGraph()
                graph.add_nodes_from(nodes)
                graph.add_edges_from(edges)
                graphs.append(graph)
                graph.graph['name'] = graph_id
                graph.graph['post_id'] = post.post_id
                graph.graph['tags'] = post.post_type
        print()
        return graphs

    def graph_labels(self, graphs):
        authors_features = []
        for i, graph in enumerate(graphs):
            print('\rextrat labels for graph {}/{}'.format(str(i + 1), len(graphs)), end='')
            if graph.graph['tags'] is not None:
                author_feature = self._create_feature('graph_labels', graph.graph['post_id'], graph.graph['tags'])
                authors_features.append(author_feature)
        print()

    def triangle_structure(self, graphs):
        authors_features = []
        for i, graph in enumerate(graphs):
            print('\rextrat triangle_structure for graph {}/{}'.format(str(i + 1), len(graphs)), end='')
            graph = nx.Graph(graph)
            all_cliques = nx.enumerate_all_cliques(graph)
            triad_cliques = [x for x in all_cliques if len(x) == 3]
            triangle_frec = len(triad_cliques) / float(len(graph.nodes))
            author_feature = self._create_feature('triangle_structure', graph.graph['post_id'], triangle_frec)
            authors_features.append(author_feature)
        return authors_features

    def in_out_degree(self, graphs):
        authors_features = []
        for i, graph in enumerate(graphs):
            print('\rextrat triangle_structure for graph {}/{}'.format(str(i + 1), len(graphs)), end='')
            in_degree = float(sum(dict(graph.in_degree()).values()))
            out_degree = float(sum(dict(graph.out_degree()).values()))
            in_out_frac = (in_degree / out_degree) / float(len(graph.nodes))
            author_feature = self._create_feature('in_out_degree', graph.graph['post_id'], in_out_frac)
            authors_features.append(author_feature)
        return authors_features
