import random
from unittest import TestCase
import networkx as nx
import pandas as pd
import numpy as np
from DB.schema_definition import DB, AuthorConnection, Author, Post
from dataset_builder.feature_extractor.sub2vec_model_creator import Sub2VecModelCreator
from dataset_builder.sub2vec_feature_generator import Sub2VecFeatureGenerator


class TestSub2VecModelCreator(TestCase):
    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestSub2VecModelCreator, cls).setUpClass()

        cls._db = DB()
        cls._db.setUp()
        cls.sub2vec_model_creator = Sub2VecModelCreator(cls._db)
        cls.sub2vec_feature_generator = Sub2VecFeatureGenerator(cls._db, **{'authors': [], 'posts': {}})

        edges = [(0, 4), (2, 0), (1, 3), (3, 1), (0, 1), (1, 2), (4, 0), (4, 3), (2, 3), (3, 0)]
        cls.connected_undirected_graph = cls.create_undirected_graph(5, edges, 'connected_undirected_graph')
        cls.unconnected_directed_graph = cls.connected_directed_graph(7, edges, 'unconnected_directed_graph')
        cls.connected_directed_graph = cls.connected_directed_graph(5, edges, 'connected_directed_graph')
        cls.unconnected_undirected_graph = cls.create_undirected_graph(7, edges, 'unconnected_undirected_graph')

        cls.add_graph_to_db(cls.connected_undirected_graph)
        cls.add_graph_to_db(cls.unconnected_directed_graph)
        cls.add_graph_to_db(cls.connected_directed_graph)
        cls.add_graph_to_db(cls.unconnected_undirected_graph)

    @classmethod
    def add_graph_to_db(cls, graph):
        post = Post(post_id=str(graph.graph['name']), domain='flickr', post_osn_id=str(graph.graph['name']))
        post.post_type = 'labels'
        author_connections = []
        for edge in graph.edges():
            author_connections.append(AuthorConnection(source_author_guid=edge[0], destination_author_guid=edge[1],
                                                       connection_type=graph.graph['name']))
        authors = []
        for node in graph.nodes():
            authors.append(Author(name=str(node), domain=str(graph.graph['name']), author_guid=str(node)))
        cls._db.addPosts([post])
        cls._db.addPosts(author_connections)
        cls._db.addPosts(authors)

    @classmethod
    def create_undirected_graph(cls, nodes_count, edges, graph_name):
        graph = nx.Graph()
        return cls.build_graph(edges, graph, graph_name, nodes_count)

    @classmethod
    def connected_directed_graph(cls, nodes_count, edges, graph_name):
        graph = nx.DiGraph()
        return cls.build_graph(edges, graph, graph_name, nodes_count)

    @classmethod
    def build_graph(cls, edges, graph, graph_name, nodes_count):
        graph.add_nodes_from(range(nodes_count))
        graph.add_edges_from(edges)
        # nx.set_node_attributes(graph, {}, 'label')
        nx.set_node_attributes(graph, values={}, name='label')
        graph.graph['name'] = graph_name
        return graph

    def setUp(self):
        random.seed(900)

    def assertArrayEquals(self, actual_vector, expected_vector):
        for actual_value, expected_value in zip(actual_vector, expected_vector):
            self.assertAlmostEqual(actual_value, expected_value, places=7)

    def test_generate_structural_embedding_for_connected_undirected_graph(self):
        args = {'dimensions': 128,
                'window': 2,
                'walkLength': 1000,
                'iterations': 20,
                'alpha': 0.5,
                'dm': 1,
                'wl_iterations': 2,
                'randomWalkCount': 10}
        embeddings = self.sub2vec_model_creator.graph_structural_embedding([self.connected_undirected_graph], **args)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 128)
        actual_vector = np.array((embeddings[0]))
        self.assertTrue(any(actual_vector))

    def test_generate_structural_embedding_for_unconnected_undirected_graph(self):
        args = {'dimensions': 138,
                'window': 2,
                'walkLength': 100,
                'iterations': 20,
                'alpha': 0.5,
                'dm': 1,
                'randomWalkCount': 10}
        embeddings = self.sub2vec_model_creator.graph_structural_embedding([self.unconnected_undirected_graph], **args)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 138)
        actual_vector = np.array((embeddings[0]))
        self.assertTrue(any(actual_vector))

    def test_generate_structural_embedding_for_connected_directed_graph(self):
        args = {'dimensions': 138,
                'window': 2,
                'walkLength': 30,
                'iterations': 20,
                'alpha': 0.5,
                'dm': 1,
                'randomWalkCount': 10}
        embeddings = self.sub2vec_model_creator.graph_structural_embedding([self.connected_directed_graph], **args)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 138)
        actual_vector = np.array((embeddings[0]))
        self.assertTrue(any(actual_vector))

    def test_generate_structural_embedding_for_unconnected_directed_graph(self):
        args = {'dimensions': 138,
                'window': 2,
                'walkLength': 40,
                'iterations': 20,
                'alpha': 0.5,
                'dm': 1,
                'randomWalkCount': 10}
        embeddings = self.sub2vec_model_creator.graph_structural_embedding([self.unconnected_directed_graph], **args)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 138)
        actual_vector = np.array((embeddings[0]))
        self.assertTrue(any(actual_vector))

    def test_generate_structural_embedding_for_4_graphs(self):
        args = {'dimensions': 118,
                'window': 2,
                'walkLength': 40,
                'iterations': 20,
                'alpha': 0.5,
                'dm': 1,
                'randomWalkCount': 10}
        graphs = [self.unconnected_directed_graph, self.connected_undirected_graph,
                  self.unconnected_undirected_graph, self.connected_directed_graph]
        embeddings = self.sub2vec_model_creator.graph_structural_embedding(graphs, **args)
        self.assertEqual(len(embeddings), 4)
        self.assertEqual(len(embeddings[0]), 118)
        self.assertEqual(len(embeddings[1]), 118)
        self.assertEqual(len(embeddings[2]), 118)
        self.assertEqual(len(embeddings[3]), 118)
        self.assertTrue(any(np.array((embeddings[0]))))
        self.assertTrue(any(np.array((embeddings[1]))))
        self.assertTrue(any(np.array((embeddings[2]))))
        self.assertTrue(any(np.array((embeddings[3]))))

    def test_generate_author_features_from_sub2vec(self):
        dimensions = 118
        args = {'dimensions': dimensions,
                'window': 2,
                'walkLength': 40,
                'iterations': 20,
                'alpha': 0.5,
                'dm': 1,
                'randomWalkCount': 10}
        graphs = [self.unconnected_directed_graph, self.connected_undirected_graph,
                  self.unconnected_undirected_graph, self.connected_directed_graph]
        embeddings = self.sub2vec_model_creator.graph_structural_embedding(graphs, **args)
        authors_features = self.sub2vec_model_creator.convert_embedding_to_author_features(graphs, embeddings)
        self.assertEqual(len(authors_features), 4 * dimensions)

        for graph, embedding in zip(graphs, embeddings):
            actual = [f.attribute_value for f in authors_features if f.author_guid == graph.graph['name']]
            self.assertArrayEquals(actual, embedding)

    def test_load_graphs(self):
        graphs = self.sub2vec_model_creator.load_graphs()
        expected_graphs = [self.unconnected_directed_graph, self.connected_undirected_graph,
                           self.unconnected_undirected_graph, self.connected_directed_graph]
        expected_graph_map = {expected_graph.graph['name']: expected_graph for expected_graph in expected_graphs}
        for actual_graph in graphs:
            expected_graph = expected_graph_map[actual_graph.graph['name']]
            self.assertNodes(actual_graph, expected_graph)
            self.assertEdges(actual_graph, expected_graph)

        pass

    def test_execute(self):
        graphs = self.sub2vec_model_creator.load_graphs()
        self.sub2vec_model_creator.execute()
        embedding_table_name = self.sub2vec_model_creator._table_name
        df = pd.read_sql_table(embedding_table_name, self._db.engine)
        self.assertTupleEqual(df.shape, (len(graphs), self.sub2vec_model_creator._num_of_dimensions + 1))

        pass

    def test_sub2vec_feature_generator(self):
        self.sub2vec_model_creator.execute()
        self.sub2vec_feature_generator.execute()

        graphs = [self.unconnected_directed_graph, self.connected_undirected_graph,
                  self.unconnected_undirected_graph, self.connected_directed_graph]
        for graph in graphs:
            actual_dimensions_count = len(self._db.get_author_features_by_author_guid(graph.graph['name']))
            self.assertEqual(actual_dimensions_count, self.sub2vec_model_creator._num_of_dimensions)


    def assertEdges(self, actual_graph, expected_graph):
        edges = [(int(v), int(u)) for v, u in actual_graph.edges()]
        self.assertListEqual(list(sorted(expected_graph.edges())), list(sorted(edges)))

    def assertNodes(self, actual_graph, expected_graph):
        self.assertListEqual(list(expected_graph.nodes()), list(sorted(map(int, actual_graph.nodes()))))
