import unittest
from DB.schema_definition import *
import datetime
from configuration.config_class import getConfig
from dataset_builder.feature_extractor.behavior_feature_generator import BehaviorFeatureGenerator

from dataset_builder.feature_extractor.key_author_score_feature_generator import KeyAuthorScoreFeatureGenerator

from dataset_builder.key_authors_model import KeyAuthorsModel
from dataset_builder.autotopic_executor import AutotopicExecutor
from dataset_builder.feature_extractor.syntax_feature_generator import SyntaxFeatureGenerator
from dataset_builder.feature_extractor.account_properties_feature_generator import AccountPropertiesFeatureGenerator
from dataset_builder.feature_extractor.link_prediction_feature_extractor import LinkPredictionFeatureExtractor


class LinkPredictionFeatureExtractorTest(unittest.TestCase):
    def setUp(self):
        self._config_parser = getConfig()
        self._domain = str(self._config_parser.get("DEFAULT", "domain"))
        self._db = DB()
        self._db.setUp()

        self._bad_actor_type = 'bad_actor'
        self._good_actor_type = 'good_actor'

        self._create_authors()
        self._graph_types = self._config_parser.eval("LinkPredictionFeatureExtractor", "graph_types")
        for type in self._graph_types:
            self._graph_type = str(type)
            self._create_connections()
        self._db.commit()

        authors = self._db.get_authors_by_domain(self._domain)
        posts = []
        parameters = {"authors": authors, "posts": posts}
        self._link_prediction_feature_extractor = LinkPredictionFeatureExtractor(self._db, **parameters)
        self._link_prediction_feature_extractor.execute()

    def tearDown(self):
        self._db.deleteDB()
        self._db.session.close()

    def testCommonNeighborsMeasure(self):

        author_feature = self._db.get_author_feature('g_1',
                                                     'link_prediction_min_common_neighbors_common_post_bad_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 1)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_max_common_neighbors_common_post_bad_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 2)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_mean_common_neighbors_common_post_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        attribute_value = round(attribute_value, 2)
        self.assertEqual(attribute_value, 1.33)
        self._db.session.close()

    def testFriendsMeasure(self):

        author_feature = self._db.get_author_feature('g_5',
                                                     'link_prediction_min_friends_measure_common_post_bad_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 1)

        author_feature = self._db.get_author_feature('g_5',
                                                     "link_prediction_max_friends_measure_common_post_bad_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 7)

        author_feature = self._db.get_author_feature('g_5',
                                                     "link_prediction_mean_friends_measure_common_post_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        self.assertEqual(attribute_value, 3.0)

        author_feature = self._db.get_author_feature('g_5',
                                                     'link_prediction_min_friends_measure_common_post_good_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 4)

        author_feature = self._db.get_author_feature('g_5',
                                                     "link_prediction_max_friends_measure_common_post_good_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 5)

        author_feature = self._db.get_author_feature('g_5',
                                                     "link_prediction_mean_friends_measure_common_post_good_actor")
        attribute_value = float(author_feature.attribute_value)
        attribute_value = round(attribute_value, 2)
        self.assertEqual(attribute_value, 4.67)
        self._db.session.close()

    def testFriendsMeasureDirectedGraph(self):

        author_feature = self._db.get_author_feature('g_5', 'link_prediction_min_friends_measure_citation_bad_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 1)

        author_feature = self._db.get_author_feature('g_5', "link_prediction_max_friends_measure_citation_bad_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 1)

        author_feature = self._db.get_author_feature('g_5', "link_prediction_mean_friends_measure_citation_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        self.assertEqual(attribute_value, 1.0)

        author_feature = self._db.get_author_feature('g_5', 'link_prediction_min_friends_measure_citation_good_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 1)

        author_feature = self._db.get_author_feature('g_5', "link_prediction_max_friends_measure_citation_good_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 2)

        author_feature = self._db.get_author_feature('g_5', "link_prediction_mean_friends_measure_citation_good_actor")
        attribute_value = float(author_feature.attribute_value)
        attribute_value = round(attribute_value, 2)
        self.assertEqual(attribute_value, 1.33)
        self._db.session.close()

    def testPreferentialAttachment(self):

        author_feature = self._db.get_author_feature('g_1',
                                                     'link_prediction_min_preferential_attachment_common_post_bad_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 6)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_max_preferential_attachment_common_post_bad_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 12)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_mean_preferential_attachment_common_post_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        self.assertEqual(attribute_value, 8.0)

        author_feature = self._db.get_author_feature('g_1',
                                                     'link_prediction_min_preferential_attachment_common_post_good_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 3)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_max_preferential_attachment_common_post_good_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 6)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_mean_preferential_attachment_common_post_good_actor")
        attribute_value = float(author_feature.attribute_value)
        self.assertEqual(attribute_value, 5.0)
        self._db.session.close()

    def testJaccard_Coefficient(self):

        author_feature = self._db.get_author_feature('g_1',
                                                     'link_prediction_min_preferential_attachment_common_post_bad_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 6)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_max_preferential_attachment_common_post_bad_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 12)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_mean_preferential_attachment_common_post_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        self.assertEqual(attribute_value, 8.0)

        author_feature = self._db.get_author_feature('g_1',
                                                     'link_prediction_min_preferential_attachment_common_post_good_actor')
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 3)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_max_preferential_attachment_common_post_good_actor")
        attribute_value = int(author_feature.attribute_value)
        self.assertEqual(attribute_value, 6)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_mean_preferential_attachment_common_post_good_actor")
        attribute_value = float(author_feature.attribute_value)
        self.assertEqual(attribute_value, 5.0)
        self._db.session.close()

    def testAdamic_Adar(self):

        author_feature = self._db.get_author_feature('g_1',
                                                     'link_prediction_min_adamic_adar_index_common_post_bad_actor')
        attribute_value = float(author_feature.attribute_value)
        attribute_value = round(attribute_value, 2)
        self.assertEqual(attribute_value, 0.72)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_max_adamic_adar_index_common_post_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        attribute_value = round(attribute_value, 2)
        self.assertEqual(attribute_value, 2.89)

        author_feature = self._db.get_author_feature('g_1',
                                                     "link_prediction_mean_adamic_adar_index_common_post_bad_actor")
        attribute_value = float(author_feature.attribute_value)
        attribute_value = round(attribute_value, 2)
        self.assertEqual(attribute_value, 1.44)
        self._db.session.close()

    # Tests that error isnt thorwn when there is no author that is classified as one of the optional_classes in the config
    # This test should print Error to the console and log
    @unittest.skip("This test is not relevant here it should test 'AnchorAuthorsCreator' module")
    def testUnclassifiedOptionalClass(self):
        test = True
        try:
            temp = self._link_prediction_feature_extractor._optional_classes
            self._link_prediction_feature_extractor._optional_classes = ["not_an_actor"]
            self._link_prediction_feature_extractor.execute()
        except:
            test = False
        self.assertTrue(test)
        self._link_prediction_feature_extractor._optional_classes = temp
        self._db.session.close()

    # Tests that error isn't thrown when there isn't any function with the same name as measure_names in the config
    # This test should print Error to the console and log
    def testIncorrectMeasureNameInConfig(self):
        test = True
        try:
            temp = self._link_prediction_feature_extractor._measure_names
            self._link_prediction_feature_extractor._measure_names = ['hi', 'common_neighbors']
            self._link_prediction_feature_extractor.execute()
        except:
            test = False
        self.assertTrue(test)
        self._link_prediction_feature_extractor._measure_names = temp
        self._db.session.close()

    # Tests that error isn't thrown when there isn't any function with the same name as aggregation_functions in the config
    # This test should print Error to the console and log
    def testIncorrectAggregationFunctionInConfig(self):
        test = True
        try:
            temp = self._link_prediction_feature_extractor._aggregation_functions_names
            self._link_prediction_feature_extractor._aggregation_functions_names = ['lol']
            self._link_prediction_feature_extractor.execute()
        except:
            test = False
        self.assertTrue(test)
        self._link_prediction_feature_extractor._aggregation_functions_names = temp
        self._db.session.close()

    def _create_authors(self):
        authors = []

        author1 = Author()
        author1.name = '1'
        author1.author_guid = 'g_1'
        author1.author_screen_name = 'TestUser1'
        author1.domain = self._domain
        author1.author_osn_id = 1
        authors.append(author1)

        author2 = Author()
        author2.name = '2'
        author2.author_guid = 'g_2'
        author2.author_screen_name = 'TestUser2'
        author2.author_type = self._bad_actor_type
        author2.domain = self._domain
        author2.author_osn_id = 2
        authors.append(author2)

        author3 = Author()
        author3.name = '3'
        author3.author_guid = 'g_3'
        author3.author_screen_name = 'TestUser3'
        author3.author_type = self._bad_actor_type
        author3.domain = self._domain
        author3.author_osn_id = 3
        authors.append(author3)

        author4 = Author()
        author4.name = '4'
        author4.author_guid = 'g_4'
        author4.author_screen_name = 'TestUser4'
        author4.author_type = self._bad_actor_type
        author4.domain = self._domain
        author4.author_osn_id = 4
        authors.append(author4)

        author5 = Author()
        author5.name = '5'
        author5.author_guid = 'g_5'
        author5.author_screen_name = 'TestUser5'
        author5.domain = self._domain
        author5.author_osn_id = 5
        authors.append(author5)

        author6 = Author()
        author6.name = '6'
        author6.author_guid = 'g_6'
        author6.author_screen_name = 'TestUser6'
        author6.author_type = self._good_actor_type
        author6.domain = self._domain
        author6.author_osn_id = 6
        authors.append(author6)

        author7 = Author()
        author7.name = '7'
        author7.author_guid = 'g_7'
        author7.author_screen_name = 'TestUser7'
        author7.author_type = self._good_actor_type
        author7.domain = self._domain
        author7.author_osn_id = 7
        authors.append(author7)

        author8 = Author()
        author8.name = '8'
        author8.author_guid = 'g_8'
        author8.author_screen_name = 'TestUser8'
        author8.author_type = self._good_actor_type
        author8.domain = self._domain
        author8.author_osn_id = 8
        authors.append(author8)

        self._db.add_authors(authors)

    def _create_connections(self):
        author_connections = []

        author_connection_1_2 = AuthorConnection()
        author_connection_1_2.source_author_guid = 'g_1'
        author_connection_1_2.destination_author_guid = 'g_2'
        author_connection_1_2.connection_type = self._graph_type
        author_connection_1_2.weight = 1.0
        author_connection_1_2.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_1_2)

        author_connection_1_3 = AuthorConnection()
        author_connection_1_3.source_author_guid = 'g_1'
        author_connection_1_3.destination_author_guid = 'g_3'
        author_connection_1_3.connection_type = self._graph_type
        author_connection_1_3.weight = 1.0
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
        author_connection_4_5.weight = 1.0
        author_connection_4_5.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_4_5)

        author_connection_5_6 = AuthorConnection()
        author_connection_5_6.source_author_guid = 'g_5'
        author_connection_5_6.destination_author_guid = 'g_6'
        author_connection_5_6.connection_type = self._graph_type
        author_connection_5_6.weight = 1.0
        author_connection_5_6.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_5_6)

        author_connection_5_7 = AuthorConnection()
        author_connection_5_7.source_author_guid = 'g_5'
        author_connection_5_7.destination_author_guid = 'g_7'
        author_connection_5_7.connection_type = self._graph_type
        author_connection_5_7.weight = 1.0
        author_connection_5_7.insertion_date = datetime.datetime.now()
        author_connections.append(author_connection_5_7)

        author_connection_5_8 = AuthorConnection()
        author_connection_5_8.source_author_guid = 'g_5'
        author_connection_5_8.destination_author_guid = 'g_8'
        author_connection_5_8.connection_type = self._graph_type
        author_connection_5_8.weight = 1.0
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
