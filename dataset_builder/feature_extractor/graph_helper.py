import networkx as nx
import logging
from configuration.config_class import getConfig


class GraphHelper:
    """
    GraphHelper provides functionality to convert a list of edges retrieved from the local database into
    a NetworkX graph object
    """
    def __init__(self, db):
        self.config_parser = getConfig()
        self._db = db
        self._targeted_classes = self.config_parser.eval("DEFAULT", "targeted_classes")

    def load_graphs(self, graphs, graph_types, graph_directed, graph_weights, authors_dict):
        """

        :param graphs: dictionary of graphs to be updated. Key is graph type, value the is NetworkX graph object
        :param graph_types: which graphs to include in the graphs object
        :param graph_directed: indicate if all graphs in this dictionary are directed
        :param graph_weights: indicate if all graphs are weighted
        :param authors_dict: dictionary of authors
        :return: updated graphs object
        """
        for graph_type in graph_types:
            if graph_type not in graphs:
                graph = self.load_graph(graph_type, graph_directed, graph_weights, authors_dict)
                if len(graph) > 0:
                    graphs[graph_type] = graph
        return graphs

    def load_graph(self, graph_type, is_directed, is_weighted, authors_dict):

            cursor = self._db.get_author_connections_by_type(str(graph_type))
            i = 0
            if cursor:
                if is_directed:
                    graph = nx.DiGraph()
                else:
                    graph = nx.Graph()
                author_connections = self._db.result_iter(cursor)
                for author_connection in author_connections:
                    source_author_guid = author_connection[0]
                    if source_author_guid in authors_dict:
                        self._add_node_to_graph(source_author_guid, graph, authors_dict)
                    destination_author_guid = author_connection[1]
                    if destination_author_guid in authors_dict:
                        self._add_node_to_graph(destination_author_guid, graph, authors_dict)

                    if source_author_guid in authors_dict and destination_author_guid in authors_dict:
                        if is_weighted:
                            weight = float(author_connection[3])
                            graph.add_edge(source_author_guid, destination_author_guid, weight=weight)
                        else:
                            graph.add_edge(source_author_guid, destination_author_guid)
                    i += 1
                logging.info("Number of author_connection of " + graph_type + " is: " + str(i))
                return graph

    def _add_node_to_graph(self, source_author_guid, graph, authors_dict):
        if not graph.has_node(source_author_guid):
            # author_type = authors_dict[source_author_guid][1]
            # author_sub_type = authors_dict[source_author_guid][2]
            author_classes_tuple = authors_dict[source_author_guid]
            i = 0
            property_dict = {}
            for targeted_class in self._targeted_classes:
                property_dict[targeted_class] = author_classes_tuple[i]
                i = i + 1
            assert isinstance(graph, nx.Graph)
            graph.add_node(source_author_guid, **property_dict)
