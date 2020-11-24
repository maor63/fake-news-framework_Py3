# Created by aviade                   
# Time: 29/03/2016 16:01

import unittest

import sys


sys.argv = ['', 'configuration/config_test.ini']

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
#
# from Twitter_API.unit_tests.twitter_api_requester_unittests import TestTwitterApiRequester
# # from preprocessing_tools.unit_tests.xml_importer_unittests import TestXmlImporter # fails 2 tests
# from DB.unit_tests.posts_unittests import TestPost
# from DB.unit_tests.authors_unittests import TestAuthor
# # from bad_actors_collector.unit_tests.bad_actor_collector_unittests import TestBadActorCollector # fails 2 tests
# from dataset_builder.unit_tests.dataset_builder_unittests import DatasetBuilderTest
# from dataset_builder.unit_tests.feature_extractor_unittests import FeatureExtractorTest
# # # LinkPredictionFeatureExtractor test fail because networkx version 2.1 has a bug need to downgrade to 1.11
# # from dataset_builder.unit_tests.link_prediction_feature_extractor_unittests import LinkPredictionFeatureExtractorTest
# # from dataset_builder.boost_authors_model import TestBA # fails 1 test
# # from missing_data_complementor.unit_tests.test_missingDataComplementor import MissingDataComplemntorTests
# from timeline_overlap_visualization.test_timelineOverlapVisualizationGenerator import TestTimelineOverlapVisualizationGenerator
# from twitter_crawler.unittests.twitter_crawler_tests import TwitterCrawlerTests
# from preprocessing_tools.unit_tests.test_app_Importer import TestAppImporter
# # from preprocessing_tools.unit_tests.test_rank_app_importer import TestRankAppImporter # fails 1 tests
# # from dataset_builder.lda_topic_model_test import TestLDATopicModel # fails 2 tests
# from dataset_builder.unit_tests.tf_idf_feature_generator_unittests import TF_IDF_Feature_Generator_Unittests
# from dataset_builder.unit_tests.n_grams_feature_generator_unittests import N_Grams_Feature_Generator_Unittests
# # missing index_field_for_prediction in section KNNWithLinkPrediction
# from experimental_environment.unittests.knn_with_link_prediction_tests import KnnTests
# from experimental_environment.unittests.knn_link_prediction_refactored_unittests import Knn_refactored_Tests
# from dataset_builder.unit_tests.click_bait_feature_generator_unittest import click_bait_feature_generator_unittest
# from dataset_builder.unit_tests.word_embedding_differential_unittests import Word_Embedding_Differential_Feature_Generator_Unittests
# # from dataset_builder.unit_tests.glove_word_embedding_model_creator_unittest import GloveWordEmbeddingModelCreatorUnittest
# from dataset_builder.unit_tests.glove_word_embeddings_feature_generator_unittests import GloveWordEmbeddingsFeatureGeneratorUnittests # fails 1 tests
# from dataset_builder.unit_tests.sentiment_feature_generator_unittest import Sentiment_Feature_Generator_Unittest
# from dataset_builder.unit_tests.text_analysis_feature_generator_unittests import Text_Analysis_Feature_Generator_Unittest
# from dataset_builder.unit_tests.known_words_number_feature_generator_unittests import Known_Words_Number_Feature_generator_Unittests
# # from preprocessing_tools.unit_tests.test_buzz_feed_politi_fact_importer import TestBuzzFeedPolitiFactImporter # fails 1 tests
# from preprocessing_tools.unit_tests.test_asonam_honeypot_importer import TestAsonamHoneypotImporter
# from preprocessing_tools.unit_tests.test_google_link_importer import TestGoogleLinkImporter
# from dataset_builder.unit_tests.test_account_properties_feature_generator import TestAccountPropertiesFeatureGenerator
# # from preprocessing_tools.unit_tests.test_reddit_crawler import TestRedditCrawler ## fails 1 tests
# from topic_distribution_visualization.test_entity_to_topic_converter import TestEntityToTopicConverter # fails 4 tests
# from dataset_builder.unit_tests.test_behaviorFeatureGenerator import TestBehaviorFeatureGenerator
# # from preprocessing_tools.unit_tests.test_instagram_crawler import TestInstagramCrawler ## fails
# from preprocessing_tools.unit_tests.test_fake_news_feature_generator import TestFakeNewsFeatureGenerator # fails 4 tests
# # from old_tweets_crawler.test_old_tweets_crawler import TestOldTweetsCrawler # fails 7 tests
# # from preprocessing_tools.unit_tests.test_fake_news_classifier import TestFakeNewsClassifier # fails 5 tests
# # from preprocessing_tools.unit_tests.test_table_to_csv_exporter import TestTableToCsvExporter # fails 2 tests
# # from preprocessing_tools.unit_tests.test_csv_to_table_importer import TestCsvToTableImporter # fails 1 test
# # from dataset_builder.unit_tests.twitter_spam_dataset_importer_unittest import Twitter_Spam_Dataset_Importer_Unittest # fails 1 tests
# from preprocessing_tools.unit_tests.test_keywords_generator import TestKeywordsGenerator # fails 3 tests
# # from dataset_builder.news_api_crawler.test_news_api_crawler import TestNewsApiCrawler # # fails 3 tests
# from dataset_builder.unit_tests.test_reddit_FeatureGenerator import RedditFeatureGeneratorTest
# from dataset_builder.unit_tests.test_behaviorFeatureGenerator import TestBehaviorFeatureGenerator
# # from dataset_builder.unit_tests.test_gensim_word_embeddings_model_trainer import TestGensimWordEmbeddingsModelTrainer # fails 2 tests
# from dataset_builder.unit_tests.tf_idf_feature_generator_unittests import TF_IDF_Feature_Generator_Unittests
# from dataset_builder.unit_tests.n_grams_feature_generator_unittests import N_Grams_Feature_Generator_Unittests
# from dataset_builder.unit_tests.feature_extractor_unittests import FeatureExtractorTest
# # from dataset_builder.unit_tests.word_embeddings_comparison_feature_generator_unittests import Word_Embeddings_Comparison_Feature_Generator_Unittests # fails 5 tests
# from dataset_builder.unit_tests.word_embedding_differential_unittests import Word_Embedding_Differential_Feature_Generator_Unittests
# from dataset_builder.unit_tests.sentiment_feature_generator_unittest import Sentiment_Feature_Generator_Unittest
# from dataset_builder.unit_tests.text_analysis_feature_generator_unittests import Text_Analysis_Feature_Generator_Unittest
# from dataset_builder.unit_tests.known_words_number_feature_generator_unittests import Known_Words_Number_Feature_generator_Unittests
# from dataset_builder.unit_tests.test_account_properties_feature_generator import TestAccountPropertiesFeatureGenerator
# from dataset_builder.unit_tests.test_syntax_feature_generator import TestSyntaxFeatureGenerator
# from dataset_builder.feature_extractor.test_aggregated_authors_posts_feature_generator import TestAggregatedAuthorsPostsFeatureGenerator
# from dataset_builder.feature_extractor.test_cooperation_topic_feature_generator import TestCooperationTopicFeatureGenerator
from dataset_builder.feature_extractor.test_temporal_feature_generator import TestTemporalFeatureGenerator # fails 1 tests
# # from preprocessing_tools.flickr_cascade_graph_builder.test_flickr_graph_builder import TestFlickrGraphBuilder
# from dataset_builder.feature_extractor.test_sub2vec_model_creator import TestSub2VecModelCreator

if __name__ == "__main__":
    unittest.main()
