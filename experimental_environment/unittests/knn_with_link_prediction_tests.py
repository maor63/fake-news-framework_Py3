import unittest
from DB.schema_definition import *
from experimental_environment.knnwithlinkprediction import KNNWithLinkPrediction
from dataset_builder.feature_extractor.link_prediction_feature_extractor import LinkPredictionFeatureExtractor
import csv

from experimental_environment.knnwithlinkprediction_refacatored import KNNWithLinkPrediction_Refactored


class KnnTests(unittest.TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._knn_algoritem = KNNWithLinkPrediction(self._db)
        self._domain = 'Microblog'
        self._knn_algoritem._domain = self._domain
        self.actor_type_dict = getConfig().eval("KNNWithLinkPrediction", "targeted_class_dict")
        self.actor_type_dict = {value: key for key, value in self.actor_type_dict.items()}
        self._bad_actor_type = 'bad_actor'
        self._good_actor_type = 'good_actor'
        self._graph_type = "cocitation"
        self._create_authors()
        self._create_connections()

    def tearDown(self):
        self._db.session.commit()
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    def test__knn_algoritem_3_neighbors(self):
        self._knn_algoritem._k = [3]
        self._knn_algoritem.execute()
        with open("data\output\KNNWithLinkPrediction\predictions_per_iteration.csv", ) as text_file:
            dr = csv.DictReader(text_file)
            for row in dr:
                iteration = eval(row['iteration'])
                user_guid = row['user_guid']
                prediction = row["prediction"]
                actual = row["actual"]
                neighbors = eval(row["neighbors"])
                neighbors = self.get_neighbor_name_and_type(neighbors)
                if (user_guid == "g_1"):
                    self.assertEqual(prediction, self._bad_actor_type)
                    self.assertTrue(('g_2', self._bad_actor_type) in neighbors)
                    self.assertTrue(('g_3', self._bad_actor_type) in neighbors)
                    self.assertTrue(('g_4', self._bad_actor_type) in neighbors)
                if (user_guid == "g_5"):
                    self.assertEqual(prediction, self._good_actor_type)
                    self.assertTrue(('g_6', self._good_actor_type) in neighbors)
                    self.assertTrue(('g_8', self._good_actor_type) in neighbors)
                    self.assertTrue(('g_4', self._bad_actor_type) in neighbors)

    def test__knn_algoritem_1_neighbors(self):
        self._knn_algoritem._k = [1]
        self._knn_algoritem.execute()

        # knn_refactors = KNNWithLinkPrediction_Refactored(self._db)
        # knn_refactors._k=[1]
        # knn_refactors.execute()
        with open("data\output\KNNWithLinkPrediction\predictions_per_iteration.csv", "r") as text_file:
            dr = csv.DictReader(text_file)
            for row in dr:
                iteration = eval(row['iteration'])
                user_guid = row['user_guid']
                prediction = row["prediction"]
                actual = row["actual"]
                neighbors = eval(row["neighbors"])
                neighbors = self.get_neighbor_name_and_type(neighbors)
                if (user_guid == "g_1"):
                    self.assertEqual(prediction, self._bad_actor_type)
                    self.assertTrue(('g_2', self._bad_actor_type) in neighbors)
                if (user_guid == "g_5"):
                    self.assertEqual(prediction, self._bad_actor_type)
                    self.assertTrue(('g_4', self._bad_actor_type) in neighbors)

    def get_neighbor_name_and_type(self, neighbors):
        return [(n[0], self.actor_type_dict[n[2]]) for n in neighbors]

    def test__knn_algoritem_more_then_max_neighbors(self):
        self._knn_algoritem._k = [8]
        self._knn_algoritem.execute()
        with open("data\output\KNNWithLinkPrediction\predictions_per_iteration.csv", "r") as text_file:
            dr = csv.DictReader(text_file)
            for row in dr:
                iteration = eval(row['iteration'])
                user_guid = row['user_guid']
                prediction = row["prediction"]
                actual = row["actual"]
                neighbors = eval(row["neighbors"])
                neighbors = self.get_neighbor_name_and_type(neighbors)
                if (user_guid == "g_1"):
                    self.assertEqual(prediction, self._bad_actor_type)
                    self.assertTrue(('g_2', self._bad_actor_type) in neighbors)
                if (user_guid == "g_5"):
                    self.assertEqual(prediction, self._good_actor_type)
                    self.assertTrue(('g_4', self._bad_actor_type) in neighbors)

    def _create_authors(self):
        authors = []

        author1 = Author()
        author1.name = '1'
        author1.domain = 'Microblog'
        author1.author_guid = 'g_1'
        author1.author_screen_name = 'TestUser1'
        author1.author_type = self._bad_actor_type
        author1.domain = self._domain
        author1.author_osn_id = 1
        authors.append(author1)

        author2 = Author()
        author2.name = '2'
        author2.domain = 'Microblog'
        author2.author_guid = 'g_2'
        author2.author_screen_name = 'TestUser2'
        author2.author_type = self._bad_actor_type
        author2.domain = self._domain
        author2.author_osn_id = 2
        authors.append(author2)

        author3 = Author()
        author3.name = '3'
        author3.domain = 'Microblog'
        author3.author_guid = 'g_3'
        author3.author_screen_name = 'TestUser3'
        author3.author_type = self._bad_actor_type
        author3.domain = self._domain
        author3.author_osn_id = 3
        authors.append(author3)

        author4 = Author()
        author4.name = '4'
        author4.domain = 'Microblog'
        author4.author_guid = 'g_4'
        author4.author_screen_name = 'TestUser4'
        author4.author_type = self._bad_actor_type
        author4.domain = self._domain
        author4.author_osn_id = 4
        authors.append(author4)

        author5 = Author()
        author5.name = '5'
        author5.domain = 'Microblog'
        author5.author_guid = 'g_5'
        author5.author_screen_name = 'TestUser5'
        author5.author_type = self._good_actor_type
        author5.domain = self._domain
        author5.author_osn_id = 5
        authors.append(author5)

        author6 = Author()
        author6.name = '6'
        author6.domain = 'Microblog'
        author6.author_guid = 'g_6'
        author6.author_screen_name = 'TestUser6'
        author6.author_type = self._good_actor_type
        author6.domain = self._domain
        author6.author_osn_id = 6
        authors.append(author6)

        author7 = Author()
        author7.name = '7'
        author7.domain = 'Microblog'
        author7.author_guid = 'g_7'
        author7.author_screen_name = 'TestUser7'
        author7.author_type = self._good_actor_type
        author7.domain = self._domain
        author7.author_osn_id = 7
        authors.append(author7)

        author8 = Author()
        author8.name = '8'
        author8.domain = 'Microblog'
        author8.author_guid = 'g_8'
        author8.author_screen_name = 'TestUser8'
        author8.author_type = self._good_actor_type
        author8.domain = self._domain
        author8.author_osn_id = 8
        authors.append(author8)

        self._db.add_authors(authors)

        lp = LinkPredictionFeatureExtractor(self._db, **{"authors": None, "posts": None})
        anchor_authors = []
        anchor_author = lp._create_anchor_author("g_2", self._bad_actor_type)
        anchor_authors.append(anchor_author)

        anchor_author = lp._create_anchor_author("g_3", self._bad_actor_type)
        anchor_authors.append(anchor_author)

        anchor_author = lp._create_anchor_author("g_4", self._bad_actor_type)
        anchor_authors.append(anchor_author)

        anchor_author = lp._create_anchor_author("g_6", self._good_actor_type)
        anchor_authors.append(anchor_author)

        anchor_author = lp._create_anchor_author("g_7", self._good_actor_type)
        anchor_authors.append(anchor_author)

        anchor_author = lp._create_anchor_author("g_8", self._good_actor_type)
        anchor_authors.append(anchor_author)

        # self._db.add_authors(anchor_authors)

    def _create_connections(self):
        author_connections = []

        author_connection_1_2 = AuthorConnection()
        author_connection_1_2.source_author_guid = 'g_1'
        author_connection_1_2.destination_author_guid = 'g_2'
        author_connection_1_2.connection_type = self._graph_type
        author_connection_1_2.weight = 2.0
        author_connection_1_2.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_1_2)

        author_connection_1_3 = AuthorConnection()
        author_connection_1_3.source_author_guid = 'g_1'
        author_connection_1_3.destination_author_guid = 'g_3'
        author_connection_1_3.connection_type = self._graph_type
        author_connection_1_3.weight = 1.3
        author_connection_1_3.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_1_3)

        author_connection_1_4 = AuthorConnection()
        author_connection_1_4.source_author_guid = 'g_1'
        author_connection_1_4.destination_author_guid = 'g_4'
        author_connection_1_4.connection_type = self._graph_type
        author_connection_1_4.weight = 1.0
        author_connection_1_4.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_1_4)

        author_connection_2_4 = AuthorConnection()
        author_connection_2_4.source_author_guid = 'g_2'
        author_connection_2_4.destination_author_guid = 'g_4'
        author_connection_2_4.connection_type = self._graph_type
        author_connection_2_4.weight = 1.0
        author_connection_2_4.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_2_4)

        author_connection_3_4 = AuthorConnection()
        author_connection_3_4.source_author_guid = 'g_3'
        author_connection_3_4.destination_author_guid = 'g_4'
        author_connection_3_4.connection_type = self._graph_type
        author_connection_3_4.weight = 1.0
        author_connection_3_4.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_3_4)

        author_connection_4_5 = AuthorConnection()
        author_connection_4_5.source_author_guid = 'g_4'
        author_connection_4_5.destination_author_guid = 'g_5'
        author_connection_4_5.connection_type = self._graph_type
        author_connection_4_5.weight = 2.0
        author_connection_4_5.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_4_5)

        author_connection_5_6 = AuthorConnection()
        author_connection_5_6.source_author_guid = 'g_5'
        author_connection_5_6.destination_author_guid = 'g_6'
        author_connection_5_6.connection_type = self._graph_type
        author_connection_5_6.weight = 1.8
        author_connection_5_6.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_5_6)

        author_connection_5_7 = AuthorConnection()
        author_connection_5_7.source_author_guid = 'g_5'
        author_connection_5_7.destination_author_guid = 'g_7'
        author_connection_5_7.connection_type = self._graph_type
        author_connection_5_7.weight = 1.5
        author_connection_5_7.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_5_7)

        author_connection_5_8 = AuthorConnection()
        author_connection_5_8.source_author_guid = 'g_5'
        author_connection_5_8.destination_author_guid = 'g_8'
        author_connection_5_8.connection_type = self._graph_type
        author_connection_5_8.weight = 1.7
        author_connection_5_8.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_5_8)

        author_connection_6_7 = AuthorConnection()
        author_connection_6_7.source_author_guid = 'g_6'
        author_connection_6_7.destination_author_guid = 'g_7'
        author_connection_6_7.connection_type = self._graph_type
        author_connection_6_7.weight = 1.0
        author_connection_6_7.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_6_7)

        self._db.add_author_connections(author_connections)
