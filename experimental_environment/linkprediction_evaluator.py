from preprocessing_tools.abstract_controller import AbstractController
import commons.commons as commons
from dataset_builder.feature_extractor.link_prediction_feature_extractor import LinkPredictionStaticFunctions
from DB.schema_definition import AuthorFeatures

import itertools
import math
import random
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd

class LinkPredictionEvaluator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._graph_types = self._config_parser.eval(self.__class__.__name__, "graph_types")
        self._number_iterations = self._config_parser.eval(self.__class__.__name__, "number_iterations")

        negative_link_method = self._config_parser.eval(self.__class__.__name__, "negative_link_method")
        percent_of_links_to_sample = self._config_parser.eval(self.__class__.__name__, "percent_of_links_to_sample")
        measure_names = self._config_parser.eval(self.__class__.__name__, "measure_names")

        self._combinations = list(itertools.product(negative_link_method, percent_of_links_to_sample, measure_names))

        self._author_features = []



    def execute(self, window_start=None):
        for graph_name, graph_type in self._graph_types.items():
            tuples = self._db.get_author_connections_by_connection_type(graph_name)
            graph = commons.create_targeted_graph(graph_type)
            commons.fill_edges_to_graph(graph, tuples)

            node_dict = graph.node
            i = 0
            # do it in order to calculate friends measure correctly.
            for author_guid, author_info_dict in node_dict.items():
                node_dict[author_guid]['community'] = i
                i += 1

            #for iteration in range(0, self._number_iterations-1):
            for negative_link_method, percent_of_links_to_sample, measure_name in self._combinations:
                 positive_links, negative_links = self.sample_links(graph, negative_link_method, percent_of_links_to_sample)
                 sampled_links = positive_links + negative_links
                 self.predict_link(graph, graph_name, sampled_links, measure_name, negative_link_method)
                 self._add_ground_truth_to_author_features(graph_name, positive_links, "Link")
                 self._add_ground_truth_to_author_features(graph_name, negative_links, "No-Link")
                 self._db.add_author_features(self._author_features)
        self._db.session.commit()

    def predict_link(self, graph, graph_name, links, measure_name, negative_edge_type):
        for link in links:
            source_node = link[0]
            dest_node = link[1]
            method_to_call = getattr(LinkPredictionStaticFunctions, measure_name)
            result = method_to_call(graph, source_node, dest_node)

            author_guid = str(graph_name + "#" + source_node + "#" + dest_node).encode('utf-8')
            attribute_name = str(graph_name + "_" + measure_name + "_" + negative_edge_type).encode('utf-8')
            af = AuthorFeatures(_author_guid=author_guid,
                                _window_start=self._window_start, _window_end=self._window_end,
                                _attribute_name=attribute_name, _attribute_value=result)
            self._author_features.append(af)

            af_2 = AuthorFeatures(_author_guid=author_guid,
                                _window_start=self._window_start, _window_end=self._window_end,
                                _attribute_name="author_screen_name", _attribute_value=author_guid)

            self._author_features.append(af_2)


    def sample_links(self, graph, negative_edge_type, percent_of_links_to_sample):
        graph_edges = graph.edges()
        num_of_graph_edges = len(graph_edges)
        number_of_links_to_sample = int(math.ceil(percent_of_links_to_sample * num_of_graph_edges))
        if (graph.number_of_edges() > number_of_links_to_sample):
            positive_edges = random.sample(graph.edges(), number_of_links_to_sample)
        else:
            positive_edges = graph.edges()

        negative_edges = []
        if negative_edge_type == 'easy':
            candidate_non_edges = list(nx.non_edges(graph))
            if len(candidate_non_edges) > number_of_links_to_sample:
                negative_edges = list(random.sample(candidate_non_edges, number_of_links_to_sample))
            else:
                negative_edges = candidate_non_edges
        else:
            matrix = nx.attr_matrix(graph)
            neighbors_distance_two = matrix[0].dot(matrix[0])
            #new_graph = nx.from_numpy_matrix(neighbors_distance_two)

            #for source, dest in new_graph.edges():
            for (source, dest), value in np.ndenumerate(neighbors_distance_two):
                source_name = matrix[1][source]
                dest_name = matrix[1][dest]
                if source == dest or value == 0  or graph.has_edge(source_name, dest_name):
                    continue
                negative_edges.append((source_name, dest_name))
                if (len(negative_edges)) == number_of_links_to_sample:
                    break


        return positive_edges, negative_edges

    def _add_ground_truth_to_author_features(self, graph_name, links, label):
        # label 1.0 = link / label 0.0 = no link

        for link in links:
            author_guid = str(graph_name + "#" + link[0] + "#" + link[1]).encode('utf-8')
            af = AuthorFeatures(_author_guid=author_guid,
                                _window_start=self._window_start, _window_end=self._window_end,
                                _attribute_name='has_link', _attribute_value=label)
            self._author_features.append(af)
