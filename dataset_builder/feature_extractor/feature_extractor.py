# Created by Jorge Bendahan (jorgeaug@post.bgu.ac.il) at 10/04/2016
# Ben Gurion University of the Neguev - Department of Information Systems Engineering

import logging
import time

from .account_properties_feature_generator import AccountPropertiesFeatureGenerator
from .behavior_feature_generator import BehaviorFeatureGenerator
from .boost_score_feature_generator import BoostScoresFeatureGenerator
from configuration.config_class import getConfig
from dataset_builder.clickbait_challenge.clickbait_feature_generator import Clickbait_Feature_Generator
from dataset_builder.feature_extractor.aggregated_authors_posts_feature_generator import \
    AggregatedAuthorsPostsFeatureGenerator
from dataset_builder.feature_extractor.claim_feature_generator import ClaimFeatureGenerator
from dataset_builder.feature_extractor.claim_graph_feature_generator import ClaimGraphFeatureGenerator
from dataset_builder.feature_extractor.claim_topic_feature_generator import ClaimTopicFeatureGenerator
from dataset_builder.feature_extractor.cooperation_topic_feature_generator import CooperationTopicFeatureGenerator
from dataset_builder.feature_extractor.flickr_feature_generator import FlickrFeatureGenerator
from dataset_builder.feature_extractor.known_words_number_feature_generator import Known_Words_Number_Feature_generator
from dataset_builder.feature_extractor.lda_topic_feature_generator import LDATopicFeatureGenerator
from dataset_builder.feature_extractor.n_gram_feature_generator import N_Grams_Feature_Generator
from dataset_builder.feature_extractor.ocr_feature_generator import OCR_Feature_Generator
from dataset_builder.feature_extractor.temporal_feature_generator import TemporalFeatureGenerator
from dataset_builder.feature_extractor.tf_idf_feature_generator import TF_IDF_Feature_Generator
from dataset_builder.feature_extractor.reddit_by_claim_post_feature_generator import RedditPostByClaimFeatureGenerator
from dataset_builder.feature_extractor.reddit_by_claim_author_feature_generator import \
    RedditAuthorByClaimFeatureGenerator
from dataset_builder.feature_extractor.glove_word_embeddings_feature_generator import \
    GloveWordEmbeddingsFeatureGenerator
from dataset_builder.feature_extractor.yelp_feature_generator import Yelp_Feature_Generator
from dataset_builder.liar_dataset.liar_dataset_feature_generator import Liar_Dataset_Feature_Generator
from dataset_builder.sub2vec_feature_generator import Sub2VecFeatureGenerator
from dataset_builder.word_embedding.word_embedding_differential_feature_generator import \
    Word_Embedding_Differential_Feature_Generator
from .distances_from_targeted_class_feature_generator import DistancesFromTargetedClassFeatureGenerator
from .graph_feature_generator import GraphFeatureGenerator
from .key_author_score_feature_generator import KeyAuthorScoreFeatureGenerator
from .link_prediction_feature_extractor import LinkPredictionFeatureExtractor
from preprocessing_tools.abstract_controller import AbstractController
from .syntax_feature_generator import SyntaxFeatureGenerator
from .topic_feature_generator import TopicFeatureGenerator
from .sentiment_feature_generator import Sentiment_Feature_Generator
from .text_analysis_feature_generator import Text_Anlalyser_Feature_Generator
from .image_tags_feature_generator import Image_Tags_Feature_Generator
from .word_embeddings_comparison_feature_generator import Word_Embeddings_Comparison_Feature_Generator

'''
This class is responsible for executing all the FeatureGenerator classes and finally calling ArffWriter class which
writes all the generated features to an output file
'''


class FeatureExtractor(AbstractController):
    def __init__(self, db, **kwargs):
        AbstractController.__init__(self, db)
        self.config_parser = getConfig()
        self._pipeline = []
        self._load_parameters = self._config_parser.eval(self.__class__.__name__, "load_parameters")

    def setUp(self):

        ###############################################################
        # MODULES
        ###############################################################
        module_names = {}

        module_names["TopicFeatureGenerator"] = TopicFeatureGenerator
        module_names["SyntaxFeatureGenerator"] = SyntaxFeatureGenerator
        module_names["BehaviorFeatureGenerator"] = BehaviorFeatureGenerator
        module_names["KeyAuthorScoreFeatureGenerator"] = KeyAuthorScoreFeatureGenerator
        module_names["BoostScoresFeatureGenerator"] = BoostScoresFeatureGenerator
        module_names["GraphFeatureGenerator_1"] = GraphFeatureGenerator
        module_names["AccountPropertiesFeatureGenerator"] = AccountPropertiesFeatureGenerator
        module_names["GraphFeatureGenerator_2"] = GraphFeatureGenerator
        module_names["DistancesFromTargetedClassFeatureGenerator"] = DistancesFromTargetedClassFeatureGenerator
        module_names["OCR_Feature_Generator"] = OCR_Feature_Generator
        module_names["Image_Tags_Feature_Generator"] = Image_Tags_Feature_Generator
        module_names["Clickbait_Feature_Generator"] = Clickbait_Feature_Generator
        module_names["Sentiment_Feature_Generator"] = Sentiment_Feature_Generator
        module_names["Text_Anlalyser_Feature_Generator"] = Text_Anlalyser_Feature_Generator
        module_names["FlickrFeatureGenerator"] = FlickrFeatureGenerator
        module_names["TF_IDF_Feature_Generator"] = TF_IDF_Feature_Generator
        module_names["N_Grams_Feature_Generator"] = N_Grams_Feature_Generator
        module_names["AggregatedAuthorsPostsFeatureGenerator"] = AggregatedAuthorsPostsFeatureGenerator
        module_names["Yelp_Feature_Generator"] = Yelp_Feature_Generator
        module_names["LDATopicFeatureGenerator"] = LDATopicFeatureGenerator
        module_names["CooperationTopicFeatureGenerator"] = CooperationTopicFeatureGenerator
        module_names["TemporalFeatureGenerator"] = TemporalFeatureGenerator

        module_names["Word_Embeddings_Comparison_Feature_Generator"] = Word_Embeddings_Comparison_Feature_Generator
        module_names["GloveWordEmbeddingsFeatureGenerator"] = GloveWordEmbeddingsFeatureGenerator
        module_names["Sub2VecFeatureGenerator"] = Sub2VecFeatureGenerator

        module_names["Word_Embedding_Differential_Feature_Generator"] = Word_Embedding_Differential_Feature_Generator
        module_names["Known_Words_Number_Feature_generator"] = Known_Words_Number_Feature_generator
        module_names["Liar_Dataset_Feature_Generator"] = Liar_Dataset_Feature_Generator
        module_names["ClaimTopicFeatureGenerator"] = ClaimTopicFeatureGenerator
        module_names["ClaimFeatureGenerator"] = ClaimFeatureGenerator
        module_names["ClaimGraphFeatureGenerator"] = ClaimGraphFeatureGenerator
        module_names["RedditPostByClaimFeatureGenerator"] = RedditPostByClaimFeatureGenerator
        module_names["RedditAuthorByClaimFeatureGenerator"] = RedditAuthorByClaimFeatureGenerator

        # LinkPredictionFeatureExtractor must be the latest. Due to the deletion of features of the anchor authors.
        module_names["LinkPredictionFeatureExtractor"] = LinkPredictionFeatureExtractor

        ###############################################################
        ## SETUP
        logging.config.fileConfig(getConfig().get("DEFAULT", "Logger_conf_file"))
        logging.info("Start Execution ... ")
        logging.info("SETUP global variables")
        window_start = getConfig().eval("DEFAULT", "start_date")
        logging.info("CREATE pipeline")

        if self._load_parameters:
            graphs = {}
            authors = self._db.get_authors()
            posts = self._db.get_author_guid_posts_dict()
            parameters = {"authors": authors, "posts": posts, "graphs": graphs}
        else:
            parameters = {"authors": [], "posts": {}, "graphs": {}}

        for module in self._config_parser.sections():
            if module_names.get(module):
                if module.startswith("GraphFeatureGenerator") or module.startswith(
                        "DistancesFromTargetedClassFeatureGenerator"):
                    self._add_graph_features_to_params(module, parameters)

                self._pipeline.append(module_names.get(module)(self._db, **parameters))

    def execute(self, window_start=None, window_end=None):
        start_time = time.time()
        info_msg = "execute started for " + self.__class__.__name__ + " started at " + str(start_time)

        print(info_msg)
        logging.info(info_msg)

        for module in self._pipeline:
            logging.info("execute module: {0}".format(module))
            T = time.time()
            logging.info('*********Started executing ' + module.__class__.__name__)

            module.execute()

            logging.info('*********Finished executing ' + module.__class__.__name__)
            T = time.time() - T

        end_time = time.time()
        diff_time = end_time - start_time
        info_msg = "execute finished for " + self.__class__.__name__ + "  in " + str(diff_time) + " seconds"
        print(info_msg)
        logging.info(info_msg)

    def _add_graph_features_to_params(self, module, parameters):

        graph_types = self._config_parser.eval(module, "graph_types")
        graph_directed = self._config_parser.eval(module, "graph_directed")
        graph_weights = self._config_parser.eval(module, "graph_weights")
        parameters.update({"graph_types": graph_types,
                           "graph_directed": graph_directed, "graph_weights": graph_weights})

        if self._config_parser.has_option(module, "algorithms"):
            algorithms = self._config_parser.eval(module, "algorithms")
            parameters.update({"algorithms": algorithms})

        if self._config_parser.has_option(module, "aggregation_functions"):
            aggregations = self._config_parser.eval(module, "aggregation_functions")
            parameters.update({"aggregation_functions": aggregations})

        if self._config_parser.has_option(module, "neighborhood_sizes"):
            neighborhood_sizes = self._config_parser.eval(module, "neighborhood_sizes")
            parameters.update({"neighborhood_sizes": neighborhood_sizes})

    def is_well_defined(self):
        # check module definition
        logging.info("start checking feature extractors definition")
        for module in self._pipeline:
            if not module.is_well_defined():
                raise Exception("module:" + module.__class__.__name__ + " config not well defined")
        return True
