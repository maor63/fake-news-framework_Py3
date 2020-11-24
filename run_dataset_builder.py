'''
Created on 26  JUN  2016

@author: Jorge Bendahan (jorgeaug@post.bgu.ac.il)


This script is responsible for performing the following tasks:
DB: set up database connection
Preprocessor: performing stemming, stop words removal (look for the Preprocessor section in config.ini
BoostAuthorModel: calculating boost author scores
AutoTopicExecutor: running LDA algorithm(topic extraction) on the crawled posts
KeyAuthorModel: calcularing key author scores
FeatureExtractor: computing features for authors and writing the final dataset of author's features into an output file.
'''
import csv
import logging.config
import os
import time
from multiprocessing import freeze_support

from DB.schema_definition import DB
from configuration.config_class import getConfig
from data_exporter.data_exporter import DataExporter
# from dataset_builder.WikiTree_Analyzer.wikitree_analyzer import WikiTreeAnalyzer
from dataset_builder.ClaimKeywordsFinder import ClaimKeywordFinder
from dataset_builder.bot_or_not.botometer_evaluator import BotometerEvaluator
from dataset_builder.classifier_hyperparameter_finder.classifier_hyperparameter_finder import \
    ClassifierHyperparameterFinder
from dataset_builder.fake_and_real_news_promoter_label_assigner.fake_and_real_news_promoter_label_assigner import \
    FakeAndRealNewsPromoterLabelAssigner
from dataset_builder.feature_extractor.sub2vec_model_creator import Sub2VecModelCreator
from dataset_builder.graph_builders.followers_graph_builder import GraphBuilder_Followers
from dataset_builder.graph_builders.friends_graph_builder import GraphBuilder_Friends
from dataset_builder.graph_builders.topic_graph_builders.topic_graph_builder_all_combinations import \
    Topic_Graph_Builder_All_Combinations
from dataset_builder.graph_builders.topic_graph_builders.topic_graph_builder_k_best import Topic_Graph_Builder_K_Best
from dataset_builder.graph_builders.topic_graph_builders.topic_graph_builder_random_combinations import \
    Topic_Graph_Builder_Random_Combinations
from dataset_builder.image_downloader.image_downloader import Image_Downloader
from dataset_builder.lda_topic_model import LDATopicModel
from dataset_builder.autotopic_executor import AutotopicExecutor
from dataset_builder.autotopic_model_creator import AutotopicModelCreator
#from dataset_builder.bag_of_words_graph_builder import GraphBuilder_Bag_Of_Words
from dataset_builder.graph_builders.bag_of_words_graph_builders.bag_of_words_graph_builder_k_best import Bag_Of_Words_Graph_Builder_K_Best
from dataset_builder.graph_builders.bag_of_words_graph_builders.bag_of_words_graph_builder_all_combinations import Bag_Of_Words_Graph_Builder_All_Combinations
from dataset_builder.graph_builders.bag_of_words_graph_builders.bag_of_words_graph_builder_random_combinations import Bag_Of_Words_Graph_Builder_Random_Combinations

from dataset_builder.boost_authors_model import BoostAuthorsModel
from dataset_builder.citation_graph_builder import GraphBuilder_Citation
from dataset_builder.clickbait_challenge.clickbait_challenge_importer import Clickbait_Challenge_Importer
from dataset_builder.cocitation_graph_builder import GraphBuilder_CoCitation
from dataset_builder.common_posts_graph_builder import GraphBuilder_Common_Posts
from dataset_builder.feature_extractor.feature_extractor import FeatureExtractor
from dataset_builder.feature_similarity_graph_builder import GraphBuilder_Feature_Similarity
from dataset_builder.key_authors_model import KeyAuthorsModel
from dataset_builder.name_analyzer.name_analyzer import NameAnalyzer

from dataset_builder.news_api_crawler.news_api_crawler import NewsApiCrawler
from dataset_builder.ocr_extractor import OCR_Extractor
from dataset_builder.randomizer.randomizer import Randomizer
from dataset_builder.topic_distribution_builder import TopicDistributionBuilder
from dataset_builder.topics_graph_builder import GraphBuilder_Topic
from dataset_builder.twitter_spam_dataset_importer import Twitter_Spam_Dataset_Importer
from dataset_builder.word_embedding.gensim_doc2vec_feature_generator import GensimDoc2VecFeatureGenerator
from preprocessing_tools.comlex_claim_tweet_importer.ComLexClaimTweetImporter import ComLexClaimTweetImporter
from preprocessing_tools.csv_to_table_importer import CsvToTableImporter
from preprocessing_tools.fake_news_snopes_importer.fake_news_snopes_importer import FakeNewsSnopesImporter
from dataset_builder.word_embedding.gensim_word_embedding_trainer import GensimWordEmbeddingsModelTrainer
from experimental_environment.clickbait_challenge_predictor import Clickbait_Challenge_Predictor
from experimental_environment.experimental_environment import ExperimentalEnvironment
from experimental_environment.experimentor import Experimentor
from experimental_environment.kernel_performance_evaluator import Kernel_Performance_Evaluator
from experimental_environment.knn_classifier import KNN_Classifier
from experimental_environment.knnwithlinkprediction import KNNWithLinkPrediction
from experimental_environment.load_datasets import Load_Datasets
from experimental_environment.predictor import Predictor
from experimental_environment.topic_authenticity_experimentor import TopicAuthenticityExperimentor
from liar_politifact_dataset_claim_updater.liar_politifact_dataset_claim_updater import \
    LiarPolitifactDatasetClaimUpdater
from missing_data_complementor.missing_data_complementor import MissingDataComplementor
#from network_graph_analyzer.network_graph_analyzer import NetworkGraphAnalyzer
from old_tweets_crawler.old_tweets_crawler import OldTweetsCrawler
from preprocessing_tools.fake_news_word_classifier.fake_news_feature_generator import FakeNewsFeatureGenerator
from preprocessing_tools.flicker_diffiusion_graph_loader import FlickrDiffusionGraphLoader
from preprocessing_tools.flickr_cascade_graph_builder.flickr_graph_builder import FlickrGraphBuilder
from preprocessing_tools.gdlet_news_importer.gdlet_news_importer import GDLET_News_Importer
from preprocessing_tools.app_importer import AppImporter
from preprocessing_tools.google_link_by_keyword_importer.google_link_importer import GoogleLinksByKeywords
from preprocessing_tools.guid_compute_updater import GuidComputeUpdater
from preprocessing_tools.instagram_crawler.instagram_crawler import InstagramCrawler
from preprocessing_tools.keywords_generator import KeywordsGenerator
from preprocessing_tools.politifact_liar_dataset_importer import Politifact_Liar_Dataset_Importer
from preprocessing_tools.politi_fact_posts_crawler.politi_fact_posts_crawler import PolitiFactPostsCrawler
from preprocessing_tools.rank_app_importer import RankAppImporter
from preprocessing_tools.create_authors_table import CreateAuthorTables
from preprocessing_tools.csv_importer import CsvImporter
from preprocessing_tools.data_preprocessor import Preprocessor
from preprocessing_tools.json_importer.json_importer import JSON_Importer
from preprocessing_tools.kaggle_importers.kaggle_fake_news_importer.kaggle_fake_news_importer import \
    Kaggle_Fake_News_Importer
from preprocessing_tools.kaggle_importers.kaggle_propoganda_importer.kaggle_propoganda_importer import \
    Kaggle_Propoganda_Importer
from preprocessing_tools.post_citation_creator import PostCitationCreator
from preprocessing_tools.fake_news_word_classifier.claim_preprocessor import ClaimPreprocessor
from preprocessing_tools.fake_news_word_classifier.fake_news_word_classifier import FakeNewsClassifier
from preprocessing_tools.reddit_crawler.reddit_crawler import RedditCrawler
from preprocessing_tools.scrapy_spiders.spider_manager import SpiderManager
from preprocessing_tools.table_duplication_remover import TableDuplicationRemover
from preprocessing_tools.table_to_csv_exporter import TableToCsvExporter
from preprocessing_tools.trec_importer.trec_2012_microblog_track_importer import Trec2012MicroblogTrackImporter

from preprocessing_tools.tumblr_importer.tumblr_importer import TumblrImporter
from preprocessing_tools.twitter_screen_names_importer.twitter_screen_names_importer import \
    TwitterAuthorScreenNamesImporter
from preprocessing_tools.us_2016_presidential_election_importer import US_2016_Presidential_Election_Importer
from preprocessing_tools.xml_importer import XMLImporter
from topic_distribution_visualization.entity_to_topic_converter import EntityToTopicConverter
from topic_distribution_visualization.topic_distribution_visualization_generator import \
    TopicDistrobutionVisualizationGenerator
from twitter_crawler.twitter_crawler import Twitter_Crawler
from experimental_environment.clickbait_challenge_evaluator import Clickbait_Challenge_Evaluator
from dataset_builder.word_embedding.glove_word_embedding_model_creator import GloveWordEmbeddingModelCreator
from dataset_builder.image_recognition.image_tags_extractor import Image_Tags_Extractor
from dataset_builder.word_embedding_graph_builder import GraphBuilder_Word_Embedding
from experimental_environment.linkprediction_evaluator import LinkPredictionEvaluator
from preprocessing_tools.buzz_feed_politi_fact_importer.buzz_feed_politi_fact_importer import BuzzFeedPolitiFactImporter
from experimental_environment.refactored_experimental_enviorment.classifier_trainer import Classifier_Trainer
from preprocessing_tools.asonam_honeypot_importer.asonam_honeypot_importer import AsonamHoneypotImporter
from dataset_builder.behind_the_name_crawler.behind_the_name_crawler import BehindTheNameCrawler
from dataset_builder.healthcare_worker_detector.healthcare_worker_detector import HealthcareWorkerDetector

###############################################################
# MODULES
###############################################################
moduleNames = {}
moduleNames["DB"] = DB  ## DB is special, it cannot be created using db.
moduleNames["XMLImporter"] = XMLImporter
moduleNames["CreateAuthorTables"] = CreateAuthorTables
moduleNames["AppImporter"] = AppImporter
moduleNames["RankAppImporter"] = RankAppImporter
moduleNames["JSON_Importer"] = JSON_Importer
moduleNames["CsvImporter"] = CsvImporter
moduleNames["Kaggle_Fake_News_Importer"] = Kaggle_Fake_News_Importer
moduleNames["Kaggle_Propoganda_Importer"] = Kaggle_Propoganda_Importer
moduleNames["TumblrImporter"] = TumblrImporter
moduleNames["Clickbait_Challenge_Importer"] = Clickbait_Challenge_Importer
moduleNames["Politifact_Liar_Dataset_Importer"] = Politifact_Liar_Dataset_Importer
moduleNames["FakeNewsSnopesImporter"] = FakeNewsSnopesImporter
moduleNames["ComLexClaimTweetImporter"] = ComLexClaimTweetImporter
moduleNames["US_2016_Presidential_Election_Importer"] = US_2016_Presidential_Election_Importer
moduleNames["TwitterAuthorScreenNamesImporter"] = TwitterAuthorScreenNamesImporter
moduleNames["BuzzFeedPolitiFactImporter"] = BuzzFeedPolitiFactImporter
moduleNames["AsonamHoneypotImporter"] = AsonamHoneypotImporter
moduleNames["Trec2012MicroblogTrackImporter"] = Trec2012MicroblogTrackImporter
moduleNames["FlickrDiffusionGraphLoader"] = FlickrDiffusionGraphLoader
moduleNames["FlickrGraphBuilder"] = FlickrGraphBuilder
moduleNames["Sub2VecModelCreator"] = Sub2VecModelCreator
moduleNames["TableToCsvExporter"] = TableToCsvExporter
moduleNames["CsvToTableImporter"] = CsvToTableImporter
moduleNames["FakeAndRealNewsPromoterLabelAssigner"] = FakeAndRealNewsPromoterLabelAssigner
moduleNames["KeywordsGenerator"] = KeywordsGenerator
moduleNames["HealthcareWorkerDetector"] = HealthcareWorkerDetector
moduleNames["TableDuplicationRemover"] = TableDuplicationRemover
moduleNames["Image_Tags_Extractor"] = Image_Tags_Extractor
moduleNames["Image_Downloader"] = Image_Downloader
moduleNames["Twitter_Crawler"] = Twitter_Crawler
moduleNames["BehindTheNameCrawler"] = BehindTheNameCrawler
moduleNames["SpiderManager"] = SpiderManager
moduleNames["MissingDataComplementor"] = MissingDataComplementor
moduleNames["GDLET_News_Importer"] = GDLET_News_Importer
moduleNames["Load_Datasets"] = Load_Datasets
moduleNames["Preprocessor"] = Preprocessor
moduleNames["EntityToTopicConverter"] = EntityToTopicConverter
moduleNames["GuidComputeUpdater"] = GuidComputeUpdater
moduleNames["BoostAuthorsModel"] = BoostAuthorsModel
moduleNames["TopicDistributionBuilder"] = TopicDistributionBuilder
moduleNames["LDATopicModel"] = LDATopicModel
moduleNames["AutotopicModelCreator"] = AutotopicModelCreator
moduleNames["AutotopicExecutor"] = AutotopicExecutor
moduleNames["OCR_Extractor"] = OCR_Extractor
moduleNames["GloveWordEmbeddingModelCreator"] = GloveWordEmbeddingModelCreator
moduleNames["GensimWordEmbeddingsModelTrainer"] = GensimWordEmbeddingsModelTrainer
moduleNames["GensimDoc2VecFeatureGenerator"] = GensimDoc2VecFeatureGenerator
moduleNames["PostCitationCreator"] = PostCitationCreator
moduleNames["KeyAuthorsModel"] = KeyAuthorsModel
moduleNames["GraphBuilder_CoCitation"] = GraphBuilder_CoCitation
moduleNames["GraphBuilder_Citation"] = GraphBuilder_Citation
moduleNames["GraphBuilder_Topic"] = GraphBuilder_Topic
moduleNames["Randomizer"] = Randomizer
moduleNames["NewsApiCrawler"] = NewsApiCrawler


moduleNames["Topic_Graph_Builder_K_Best"] = Topic_Graph_Builder_K_Best
moduleNames["Topic_Graph_Builder_All_Combinations"] = Topic_Graph_Builder_All_Combinations
moduleNames["Topic_Graph_Builder_Random_Combinations"] = Topic_Graph_Builder_Random_Combinations
#moduleNames["GraphBuilder_Bag_Of_Words"] = GraphBuilder_Bag_Of_Words

moduleNames["Bag_Of_Words_Graph_Builder_K_Best"] = Bag_Of_Words_Graph_Builder_K_Best
moduleNames["Bag_Of_Words_Graph_Builder_All_Combinations"] = Bag_Of_Words_Graph_Builder_All_Combinations
moduleNames["Bag_Of_Words_Graph_Builder_Random_Combinations"] = Bag_Of_Words_Graph_Builder_Random_Combinations

moduleNames["GraphBuilder_Word_Embedding"] = GraphBuilder_Word_Embedding
moduleNames["GraphBuilder_Common_Posts"] = GraphBuilder_Common_Posts
moduleNames["GraphBuilder_Feature_Similarity"] = GraphBuilder_Feature_Similarity
moduleNames["GraphBuilder_Followers"] = GraphBuilder_Followers
moduleNames["GraphBuilder_Friends"] = GraphBuilder_Friends
moduleNames["FeatureExtractor"] = FeatureExtractor
moduleNames["DataExporter"] = DataExporter
moduleNames["LinkPredictionEvaluator"] = LinkPredictionEvaluator
moduleNames["ExperimentalEnvironment"] = ExperimentalEnvironment
moduleNames["Clickbait_Challenge_Evaluator"] = Clickbait_Challenge_Evaluator
moduleNames["Predictor"] = Predictor
moduleNames["Clickbait_Challenge_Predictor"] = Clickbait_Challenge_Predictor
moduleNames["ClaimPreprocessor"] = ClaimPreprocessor
moduleNames["FakeNewsClassifier"] = FakeNewsClassifier
moduleNames["FakeNewsFeatureGenerator"] = FakeNewsFeatureGenerator

moduleNames["KNNWithLinkPrediction"] = KNNWithLinkPrediction
moduleNames["KNN_Classifier"] = KNN_Classifier
moduleNames["BotometerEvaluator"] = BotometerEvaluator
moduleNames["Kernel_Performance_Evaluator"] = Kernel_Performance_Evaluator
moduleNames["TopicDistrobutionVisualizationGenerator"] = TopicDistrobutionVisualizationGenerator
moduleNames["TopicAuthenticityExperimentor"] = TopicAuthenticityExperimentor
moduleNames["Politifact_Posts_Importer"] = PolitiFactPostsCrawler
moduleNames["Twitter_Spam_Dataset_Importer"] = Twitter_Spam_Dataset_Importer
moduleNames["ClaimKeywordFinder"] = ClaimKeywordFinder
moduleNames["OldTweetsCrawler"] = OldTweetsCrawler
moduleNames["RedditCrawler"] = RedditCrawler
moduleNames["GoogleLinksByKeywords"] = GoogleLinksByKeywords
moduleNames["InstagramCrawler"] = InstagramCrawler

moduleNames["LiarPolitifactDatasetClaimUpdater"] = LiarPolitifactDatasetClaimUpdater
#moduleNames["NetworkGraphAnalyzer"] = NetworkGraphAnalyzer
# moduleNames["WikiTreeAnalyzer"] = WikiTreeAnalyzer
moduleNames["NameAnalyzer"] = NameAnalyzer
moduleNames["Classifier_Trainer"] = Classifier_Trainer
moduleNames["ClassifierHyperparameterFinder"] = ClassifierHyperparameterFinder
moduleNames["Experimentor"] = Experimentor

###############################################################
## SETUP

logging.config.fileConfig(getConfig().get("DEFAULT", "Logger_conf_file"))
config = getConfig()
domain = str(config.get("DEFAULT", "domain"))
logging.info("Start Execution ... ")
logging.info("SETUP global variables")

window_start = getConfig().eval("DEFAULT", "start_date")
newbmrk = os.path.isfile("benchmark.csv")
bmrk_file = open("benchmark.csv", "a")
bmrk_results = csv.DictWriter(bmrk_file,
                              ["time", "jobnumber", "config", "window_size", "window_start", "dones", "posts",
                               "authors"] + list(moduleNames.keys()),
                              dialect="excel", lineterminator="\n")

if not newbmrk:
    bmrk_results.writeheader()

logging.info("CREATE pipeline")
db = DB()
moduleNames["DB"] = lambda x: x
pipeline = []
for module in getConfig().sections():
    parameters = {}
    if moduleNames.get(module):
        pipeline.append(moduleNames.get(module)(db))

logging.info("SETUP pipeline")
bmrk = {"config": getConfig().getfilename(), "window_start": "setup"}

for module in pipeline:
    logging.info("setup module: {0}".format(module))
    T = time.time()
    module.setUp()
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()
#
# clean_authors_features = getConfig().eval("DatasetBuilderConfig", "clean_authors_features_table")
# if clean_authors_features:
#     db.delete_authors_features()

clean_authors_features = getConfig().eval("DatasetBuilderConfig", "clean_authors_features_table")
if clean_authors_features:
    db.delete_authors_features()

#check defenition
logging.info("checking module definition")
for module in pipeline:
    if not module.is_well_defined():
        raise Exception("module: "+ module.__class__.__name__ +" config not well defined")
    logging.info("module "+str(module) + " is well defined")

###############################################################
## EXECUTE
bmrk = {"config": getConfig().getfilename(), "window_start": "execute"}
for module in pipeline:
    logging.info("execute module: {0}".format(module))
    T = time.time()
    logging.info('*********Started executing ' + module.__class__.__name__)

    module.execute(window_start)

    logging.info('*********Finished executing ' + module.__class__.__name__)
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

num_of_authors = db.get_number_of_targeted_osn_authors(domain)
bmrk["authors"] = num_of_authors

num_of_posts = db.get_number_of_targeted_osn_posts(domain)
bmrk["posts"] = num_of_posts

bmrk_results.writerow(bmrk)
bmrk_file.flush()

# TearDown
for module in pipeline:
    logging.info("tear down module: {0}".format(module))
    T = time.time()
    logging.info('*********Started executing ' + module.__class__.__name__)

    module.tearDown()

if __name__ == '__main__':

    pass
