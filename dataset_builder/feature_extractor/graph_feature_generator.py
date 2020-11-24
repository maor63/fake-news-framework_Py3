# Created by jorgeaug at 30/06/2016

import networkx as nx

from configuration.config_class import getConfig
import pandas as pd
from commons.consts import Algorithms
import math
from commons.commons import *
import sys
from .graph_helper import GraphHelper
from DB.schema_definition import AuthorFeatures


class GraphFeatureGenerator:
    def __init__(self, db, **kwargs):
        self.config_parser = getConfig()
        start_date = self.config_parser.get("DEFAULT", "start_date").strip("date('')")
        self._window_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self._window_size = datetime.timedelta(seconds=int(self.config_parser.get("DEFAULT", "window_analyze_size_in_sec")))
        self._window_end = self._window_start + self._window_size
        self._db = db

        self._targeted_classes = self.config_parser.eval("DEFAULT", "targeted_classes")

        if 'authors' in kwargs:
            self._authors = kwargs['authors']
            self._author_dict = self._create_author_dictionary(self._authors)
        else:
            raise Exception('Author object was not passed as parameter')

        if kwargs.keys() >= {'graph_types', 'algorithms', 'aggregation_functions', 'neighborhood_sizes', 'graph_weights', 'graph_directed'}:
            self._graph_types = kwargs['graph_types']
            self._algorithms = kwargs['algorithms']
            self._aggregations_functions = kwargs['aggregation_functions']
            self._neighborhood_sizes = kwargs['neighborhood_sizes']
            self._graph_weights = kwargs['graph_weights']
            self._graph_directed = kwargs['graph_directed']
            self._graphs = kwargs["graphs"]
            self.load_graphs()
        else:
            raise Exception('Graph parameters for feature generation are missing or incomplete')

    def load_graphs(self):
        graph_helper = GraphHelper(self._db)
        self._graphs = graph_helper.load_graphs(self._graphs, self._graph_types, self._graph_directed,
                                                self._graph_weights, self._author_dict)

    def execute(self):

        info_msg = "executing " + self.__class__.__name__
        logging.info(info_msg)
        total_authors = len(self._authors)
        authors_features = []

        for graph_name, graph in list(self._graphs.items()):
            info_msg = "processing " + graph_name + " graph "
            logging.info(info_msg)

            #if the current graph does not match the configuration, we skip
            if self._graph_directed != nx.is_directed(graph):
                continue

            for algorithm in self._algorithms:
                if nx.is_directed(graph) and algorithm == Algorithms.CLUSTERING:
                    continue
                elif not nx.is_directed(graph) and (algorithm == Algorithms.IN_DEGREE_CENTRALITY
                                                   or algorithm == Algorithms.OUT_DEGREE_CENTRALITY):
                    continue

                info_msg = "calculating " + algorithm
                logging.info(info_msg)
                method_to_call = getattr(nx, algorithm)
                result = method_to_call(graph)
                processed_authors = 0
                for author in self._authors:
                    """ Per-node features: in_degree, out_degree, it+out degree, betweneess, closeness, local clustering coefficient of each author (node)"""
                    author_guid = author.author_guid
                    if author_guid in result:
                        attribute_name = str(graph_name+"_net_"+algorithm)
                        attribute_value = result[author_guid]

                        author_feature = self._db.create_author_feature(author_guid, attribute_name, attribute_value)
                        authors_features.append(author_feature)

                        """ Aggregated function features: mean, std dev, skewness and kurtosis of scores of neighboring nodes"""
                        if self._aggregations_functions and self._neighborhood_sizes:
                            for neighborhood_size in self._neighborhood_sizes:
                                for aggregation_f in self._aggregations_functions:
                                    neighbors = nx.single_source_shortest_path_length(graph, author_guid, cutoff=neighborhood_size)
                                    if len(neighbors) > 0:
                                        neighbors_score = pd.Series(data=[result[x] for x in neighbors])
                                        method_to_call = getattr(neighbors_score, aggregation_f)
                                        value = method_to_call()
                                        if math.isnan(value):
                                            #value = None
                                            continue

                                        attribute_name = str(
                                            graph_name + "_net_" + algorithm + "_aggregation_function_" + aggregation_f + "_neighb_size_" + str(
                                                neighborhood_size))
                                        attribute_value = str(value)
                                        author_feature = self._db.create_author_feature(author_guid, attribute_name, attribute_value)
                                        authors_features.append(author_feature)

                    processed_authors += 1
                    info_msg = "\r "+graph_name+" graph: processing author " + str(processed_authors) + " from " + str(total_authors)
                    print (info_msg, end="")

        if authors_features:
            for author_features_row in authors_features:
                self._db.update_author_features(author_features_row)
            self._db.commit()

    '''
        There is the same function in commons.
        However, we get exception TypeError: create_author_dictionary() takes exactly 2 arguments (1 given)
        Need to understand why!!
    '''

    def _create_author_dictionary(self, authors):
        """
        Converts a list of Authors objects into a dictionary.
        The dictionary's key is author_guid
        The value is a tuple containing Author Type, Sub Type
        :param authors:
        :return: dictionary of authors
        """
        author_dictionary = {}
        for author in authors:
            author_guid = author.author_guid
            #author_osn_id = getattr(author, "author_osn_id")
            tuple = ()
            #tuple = tuple + (author_osn_id,)

            for targeted_class in self._targeted_classes:
                targeted_class_value = getattr(author, targeted_class)

                tuple = tuple + (targeted_class_value,)

            author_dictionary[author_guid] = tuple
        return author_dictionary

    def is_well_defined(self):
        return True
