# Created by aviade at 23/01/2017



from types import FunctionType

import pandas as pd
from .base_feature_generator import BaseFeatureGenerator
import networkx as nx
import math
import sys
import logging
from dataset_builder.feature_extractor.anchor_authors_creator import AnchorAuthorsCreator


class LinkPredictionFeatureExtractor(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)

        self._measure_names = self._config_parser.eval(self.__class__.__name__, "measure_names")
        self._aggregation_functions_names = self._config_parser.eval(self.__class__.__name__, "aggregation_functions")
        self._graph_types = self._config_parser.eval(self.__class__.__name__, "graph_types")
        self._property_node_field_names = self._config_parser.eval(self.__class__.__name__, "property_node_field_names")
        self._anchor_authors_creator = AnchorAuthorsCreator(self._db)

    def execute(self, window_start=None):
        anchor_authors_creator = self._anchor_authors_creator

        targeted_class_anchor_author_dict, anchor_authors_dict = anchor_authors_creator.create_anchor_author_dictionary()
        for graph_type in self._graph_types:
            try:
                self._edges = self._db.get_author_connections_by_connection_type(graph_type)

                author_guid_targeted_classes_tuple_dict = self._create_author_guid_targeted_classes_tuple_dict()
                graph = self._create_targeted_graph(author_guid_targeted_classes_tuple_dict,
                                                    self._graph_types[graph_type])
                if (graph is None):
                    break
                node_dict = graph.node
                i = 0
                # do it in order to calculate friends measure correctly.
                for author_guid, author_info_dict in node_dict.items():
                    node_dict[author_guid]['community'] = i
                    i += 1

                for measure_name in self._measure_names:
                    self._calculate_link_prediction_features(measure_name, graph_type, graph, node_dict,
                                                                   targeted_class_anchor_author_dict,
                                                                   anchor_authors_dict)

            except AttributeError as e:
                if (e[0] == "measure_name"):
                    print('Error: {0} in invalid measure name'.format(e[1]), file=sys.stderr)
                    logging.error('Error: {0} in invalid measure name'.format(e[1]))
                elif (e[0] == "aggregation_function"):
                    print('Error: {0} in invalid aggregation function'.format(e[1]), file=sys.stderr)
                    logging.error('Error: {0} in invalid aggregation function'.format(e[1]))

        self._db.delete_anchor_author_features()

    def _create_author_guid_targeted_classes_tuple_dict(self):
        author_guid_targeted_classes_tuple_dict = {}
        for author in self.authors:
            author_guid = author.author_guid
            tuple = ()
            for property_node_field_name in self._property_node_field_names:
                targeted_class = getattr(author, property_node_field_name)
                tuple = tuple + (targeted_class,)

            author_guid_targeted_classes_tuple_dict[author_guid] = tuple
        return author_guid_targeted_classes_tuple_dict

    def _create_targeted_graph(self, author_guid_targeted_classes_tuple_dict, graph_type):
        if (graph_type == 'undirected'):
            graph = nx.Graph()
        elif (graph_type == 'directed'):
            graph = nx.DiGraph()
        else:
            print("Graph type has to be directed or undirected")
            return None
        self._fill_edges_to_graph(graph)
        self._fill_nodes_properties_to_graph(author_guid_targeted_classes_tuple_dict, graph)

        return graph

    # doesn't work- dont know if needed
    # def weighted_jaccard_coefficient_fuzzy_sets(self, graph, author_osn_id, labeled_author_osn_id):
    #     return 0
    #     numerator = 0.0
    #     denominator = 0.0
    #     for z in nx.common_neighbors(graph, author_osn_id, labeled_author_osn_id):
    #         if graph.get_edge_data(author_osn_id, z)['weight'] > graph.get_edge_data(labeled_author_osn_id, z)[
    #             'weight']:
    #             numerator += graph.get_edge_data(labeled_author_osn_id, z)['weight']
    #         else:
    #             numerator += graph.get_edge_data(author_osn_id, z)['weight']
    #
    #     total_neighbors = nx.neighbors(graph, author_osn_id) + nx.neighbors(graph, labeled_author_osn_id)
    #     for neighbor in total_neighbors:
    #         edge_a = graph.get_edge_data(author_osn_id, neighbor, default=0)
    #         edge_b = graph.get_edge_data(labeled_author_osn_id, neighbor, default=0)
    #
    #         if edge_a != 0 and edge_b != 0:
    #             if edge_a['weight'] > edge_b['weight']:
    #                 denominator += edge_a['weight']
    #             else:
    #                 denominator += edge_b['weight']
    #         elif edge_a != 0:
    #             denominator += edge_a['weight']
    #         elif edge_b != 0:
    #             denominator += edge_b['weight']
    #
    #     if denominator > 0:
    #         return numerator / denominator
    #     else:
    #         return 0

    def _fill_edges_to_graph(self, graph):
        edges = []
        for tuple in self._edges:
            source_author_guid = tuple[0]
            dest_author_guid = tuple[1]
            weight = tuple[2]

            edges.append((source_author_guid, dest_author_guid, {"weight": weight}))

        graph.add_edges_from(edges)

    def _fill_nodes_properties_to_graph(self, author_guid_targeted_classes_tuple_dict, graph):
        nodes = []
        for author_guid, tuple in author_guid_targeted_classes_tuple_dict.items():
            properties_dict = {}
            i = 0
            for property_node_field_name in self._property_node_field_names:
                properties_dict[property_node_field_name] = tuple[i]
                i += 1

            property_tuple = (author_guid, properties_dict)
            nodes.append(property_tuple)

        graph.add_nodes_from(nodes)

    def _calculate_link_prediction_features(self, measure_name, graph_type, graph, node_dict,
                                            targeted_class_anchor_author_dict, anchor_authors_dict):
        if measure_name == "bayesian_promising":
            LinkPredictionStaticFunctions.bayesian_promising(measure_name, graph_type, graph, node_dict, targeted_class_anchor_author_dict, anchor_authors_dict, self._db)
            return
        author_features = []
        for author_guid, author_info_dict in node_dict.items():
            if author_guid in anchor_authors_dict:
                continue
            for targeted_class, anchor_author_dict in targeted_class_anchor_author_dict.items():
                author_scores = []
                for anchor_author_guid, anchor_author_guid in anchor_author_dict.items():
                    if anchor_author_guid in node_dict:
                        author_score = getattr(LinkPredictionStaticFunctions, measure_name)(graph, author_guid, anchor_author_guid)
                        if author_score != 0:
                            author_scores.append(author_score)
                if len(author_scores) > 0:
                    for aggregation_function_name in self._aggregation_functions_names:
                        author_scores_series = pd.Series(data=author_scores)
                        try:
                            final_author_score = getattr(author_scores_series, aggregation_function_name)()
                            if math.isnan(final_author_score):
                                continue

                            attribute_name = 'link_prediction_' + aggregation_function_name + "_" + measure_name + "_" + graph_type + "_" + targeted_class
                            author_feature = self._db.create_author_feature(author_guid, str(attribute_name),
                                                                            final_author_score)

                            author_features.append(author_feature)
                        except:
                            raise AttributeError("aggregation_function", aggregation_function_name)
        self._db.save_author_features(author_features)

    def is_well_defined(self):
        requested_set = self._measure_names
        has_set = LinkPredictionStaticFunctions.get_all_functions()
        flag = True
        for func in requested_set:
            if func not in has_set:
                raise Exception("function "+func +" is not defined in this module: "+self.__class__.__name__)
                flag = False
        return flag

    def _create_anchor_author(self, author_guid, author_type):
        self._db.insert_anchor_author(author_guid,author_type)

class LinkPredictionStaticFunctions():
    @staticmethod
    def friends_measure(graph, author_guid, labeled_author_guid):
        friends_measure_score = 0
        author_guid_neighbors = list(nx.all_neighbors(graph, author_guid))
        labeled_author_osn_id_neighbors = list(nx.all_neighbors(graph, labeled_author_guid))
        dict_author_guid_neighbors_a = {}
        for author in author_guid_neighbors:
            dict_author_guid_neighbors_a[author] = ""

        dict_labeled_author_guid_neighbors_b = {}
        for author in labeled_author_osn_id_neighbors:
            dict_labeled_author_guid_neighbors_b[author] = ""

        all_edges_dict = graph.edge

        for side_a in all_edges_dict:
            neighbor_a = side_a
            for side_b in all_edges_dict[side_a]:
                neighbor_b = side_b
                if neighbor_a != neighbor_b:
                    if (
                            neighbor_a in dict_author_guid_neighbors_a and neighbor_b in dict_labeled_author_guid_neighbors_b):
                        friends_measure_score += 1
                else:  # it means that author_osn_id and labeled_author_osn_id hold the same neighbor.
                    friends_measure_score += 1

        return friends_measure_score

    @staticmethod
    def weighted_adamic_addar(graph, author_osn_id, labeled_author_osn_id):
        weighted_adamic_addar_score = 0.0
        for z in nx.common_neighbors(graph, author_osn_id, labeled_author_osn_id):
            numerator = graph.get_edge_data(author_osn_id, z)['weight'] + graph.get_edge_data(labeled_author_osn_id, z)[
                'weight']
            inner_sum = 0.0
            for c in nx.neighbors(graph, z):
                inner_sum += graph.get_edge_data(z, c)['weight']
            denominator = math.log(1 + inner_sum)
            if denominator > 0.0:
                weighted_adamic_addar_score += numerator / denominator
        return weighted_adamic_addar_score

    @staticmethod
    def common_friends(graph, author_osn_id, labeled_author_osn_id):
        common_neighbors_iterator = nx.common_neighbors(graph, author_osn_id, labeled_author_osn_id)
        common_neighbors = list(common_neighbors_iterator)
        common_neighbors_count = len(common_neighbors)
        return common_neighbors_count

    @staticmethod
    def common_neighbors(graph, author_osn_id, labeled_author_osn_id):
        if not nx.is_directed(graph):
            pair = [(author_osn_id, labeled_author_osn_id)]
            common_neighbors_iterator = nx.nx.cn_soundarajan_hopcroft(graph, pair)
            common_neighbors_score = LinkPredictionStaticFunctions.get_score_from_iterator(common_neighbors_iterator)
            return common_neighbors_score
        return 0

    @staticmethod
    def preferential_attachment(graph, author_osn_id, labeled_author_osn_id):
        if not nx.is_directed(graph):
            pair = [(author_osn_id, labeled_author_osn_id)]
            preferential_attachment_iterator = nx.preferential_attachment(graph, pair)
            preferential_attachment_score = LinkPredictionStaticFunctions.get_score_from_iterator(preferential_attachment_iterator)
            return preferential_attachment_score
        return 0

    @staticmethod
    def total_friends(graph, author_guid, labeled_author_guid):
        num_of_neighbors_author_guid = len(list(nx.all_neighbors(graph, author_guid)))
        num_of_neighbors_labeled_author_guid = len(list(nx.all_neighbors(graph, labeled_author_guid)))

        total = num_of_neighbors_author_guid + num_of_neighbors_labeled_author_guid
        return total

    @staticmethod
    def jaccard_coefficient(graph, author_osn_id, labeled_author_osn_id):
        if not nx.is_directed(graph):
            pair = [(author_osn_id, labeled_author_osn_id)]
            jaccard_coefficient_iterator = nx.jaccard_coefficient(graph, pair)
            jaccard_coefficient_score = LinkPredictionStaticFunctions.get_score_from_iterator(jaccard_coefficient_iterator)
            return jaccard_coefficient_score
        return 0
    @staticmethod
    def adamic_adar_index(graph, author_osn_id, labeled_author_osn_id):
        if not nx.is_directed(graph):
            pair = [(author_osn_id, labeled_author_osn_id)]
            adamic_adar_iterator = nx.adamic_adar_index(graph, pair)
            adamic_adar_score = LinkPredictionStaticFunctions.get_score_from_iterator(adamic_adar_iterator)
            return adamic_adar_score
        return 0

    @staticmethod
    def transitive_friends_v1(graph, author_guid, labeled_author_guid):
        intersection = 0
        if nx.is_directed(graph):
            num_of_out_degree_author_guid_neighbors = graph.out_degree(author_guid)
            num_of_in_degree_labeled_author_guid_neighbors = graph.in_degree(labeled_author_guid)
            intersection = min(num_of_out_degree_author_guid_neighbors,num_of_in_degree_labeled_author_guid_neighbors)
        return intersection

    @staticmethod
    def transitive_friends_v2(graph, author_guid, labeled_author_guid):
        intersection = 0
        if nx.is_directed(graph):
            num_of_out_degree_labeled_author_guid_neighbors = graph.out_degree(labeled_author_guid)
            num_of_in_degree_author_guid_neighbors = graph.in_degree(author_guid)
            intersection = min(num_of_out_degree_labeled_author_guid_neighbors, num_of_in_degree_author_guid_neighbors)
        return intersection

    @staticmethod
    def opposite_direction_friends(graph, author_guid, labeled_author_guid):
        'The function opposite_direction_friends(v,u) ideicates whether reciprocal connections exist between vertices (v, u)'
        'It returns 0 whether the graph is undirected.'
        'Else if there is no connection it should return 1 else 2'
        opposite = 0
        if nx.is_directed(graph):
            opposite = 1
            if graph.has_edge(labeled_author_guid, author_guid):
                opposite = 2
        return opposite

    @staticmethod
    def jaccard_coefficient_weighted_sum(graph, author_osn_id, labeled_author_osn_id):
        weighted__jaccard_coefficient = 0.0
        denominator = graph.degree(author_osn_id, 'weight') + graph.degree(labeled_author_osn_id, 'weight')
        if denominator > 0.0:
            for z in nx.common_neighbors(graph, author_osn_id, labeled_author_osn_id):
                numerator = graph.get_edge_data(author_osn_id, z)['weight'] + \
                            graph.get_edge_data(labeled_author_osn_id, z)['weight']
                weighted__jaccard_coefficient += numerator / denominator

        return weighted__jaccard_coefficient

    @staticmethod
    def bayesian_promising(measure_name, graph_type, graph, node_dict,
                                              targeted_class_anchor_author_dict, anchor_authors_dict, db):
        author_features = []
        for author_guid, author_info_dict in node_dict.items():
            if author_guid in anchor_authors_dict:
                continue
            for targeted_class, anchor_author_dict in targeted_class_anchor_author_dict.items():

                author_guid_leads, total_neighbors = LinkPredictionStaticFunctions.find_leads_connected_to_author(author_guid, graph,
                                                                                          targeted_class,
                                                                                          targeted_class_anchor_author_dict)

                likelihoods = []
                for lead in author_guid_leads:
                    lead_leads, total_neighbors = LinkPredictionStaticFunctions.find_leads_connected_to_author(lead, graph, targeted_class,
                                                                                       targeted_class_anchor_author_dict)
                    num_of_leads = len(lead_leads)
                    lead_promising_factor = num_of_leads / total_neighbors
                    # likelihood that a friend of m is not a lead
                    likelihood_not_lead = 1 - lead_promising_factor
                    likelihoods.append(likelihood_not_lead)

                multipication = 0
                for likelihood in likelihoods:
                    multipication *= likelihood

                bayesian_promising_score = 1 - multipication
                attribute_name = 'link_prediction_' + measure_name + "_" + graph_type + "_" + targeted_class
                author_feature = db.create_author_feature(author_guid, str(attribute_name),
                                                                bayesian_promising_score)

                author_features.append(author_feature)

        db.save_author_features(author_features)

    @staticmethod
    def find_leads_in_neighbors(targeted_class, targeted_class_anchor_author_dict, author_guid_neighbors):
        leads = []
        total_neighbors = 0
        for neighbor in author_guid_neighbors:
            total_neighbors += 1
            if neighbor in targeted_class_anchor_author_dict[targeted_class]:
                leads.append(neighbor)
        return leads, total_neighbors

    @staticmethod
    def find_leads_connected_to_author(author_guid, graph, targeted_class, targeted_class_anchor_author_dict):
        author_guid_neighbors = nx.all_neighbors(graph, author_guid)

        leads, total_neighbors = LinkPredictionStaticFunctions.find_leads_in_neighbors(targeted_class, targeted_class_anchor_author_dict,
                                                           author_guid_neighbors)
        return leads, total_neighbors

    @staticmethod
    def get_score_from_iterator(measure_iterator):
        tuple = list(measure_iterator)
        score = tuple[0][2]
        return score

    @staticmethod
    def get_all_functions():
        return set([m for m in dir(LinkPredictionStaticFunctions) if callable(getattr(LinkPredictionStaticFunctions, m))])
