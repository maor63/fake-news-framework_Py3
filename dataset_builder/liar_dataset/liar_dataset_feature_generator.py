

from DB.schema_definition import AuthorFeatures, Politifact_Liar_Dataset
from commons.commons import *
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
import nltk
from nltk import word_tokenize
from nltk.tag import map_tag
from nltk.corpus import words
from nltk.corpus import stopwords

import re

from preprocessing_tools.abstract_controller import AbstractController


class Liar_Dataset_Feature_Generator(AbstractController):
    def __init__(self, db, **kwargs):
        AbstractController.__init__(self, db)
        attributes_to_remove = ['post_guid','original_id','statement','targeted_label','metadata','false_count','barely_true_count', 'half_true_count','mostly_true_count','pants_on_fire']
        self._attribute_names = self._get_requested_attributes(attributes_to_remove)
        self._counter = 0

    def execute(self, window_start= None):
        self._liar_dataset_records = self._db.get_liar_dataset_records()
        features = self._liar_dataset_records_to_author_features(self._liar_dataset_records)
        self._db.add_author_features(features)
        self._db.commit()
        # features_batches = [features[x:x+100] for x in xrange(0, len(features), 100)]
        # for batch in features_batches:
        #     self._db.add_author_features(batch)
        #     self._db.commit()
        # logging.info("finished createing "+str(len(features))+ " features")
        # self._db.update_author_features(features)
        # self._db.commit()
        # subject_to_number_dict = {}
        # subject_counter = 0
        # for liar_dataset_record in self._liar_dataset_records:
        #     subject_record = liar_dataset_record.subject
        #     subjects = subject_record.split(",")
        #     for subject in subjects:
        #         if subject not in subject_to_number_dict:
        #             subject_to_number_dict[subject] = subject_counter
        #             subject_counter += 1


    def _liar_dataset_records_to_author_features(self, liar_records):
        author_features = []
        af = AuthorFeatures()
        for attribute_name in self._attribute_names :
            logging.info("starting proccessing attribute: "+str(attribute_name))
            author_features +=([ self._parase_singel_line(line, attribute_name) for line in liar_records])
        return author_features

    def _parase_singel_line(self, line, attribute_name):
        af = AuthorFeatures()
        af.author_guid = str(getattr(line, 'post_guid'))
        af.attribute_name = str(attribute_name)
        af.attribute_value = str(getattr(line,attribute_name))
        af.window_start = self._window_start
        af.window_end = self._window_end
        return af

    def _get_requested_attributes(self, attributes_to_remove):
        attribute_names = [a for a in dir(Politifact_Liar_Dataset) if not a.startswith('__') and not callable(getattr(Politifact_Liar_Dataset,a)) and not a.startswith('_')]
        attribute_names = list([attribute_name for attribute_name in attribute_names if attribute_name not in attributes_to_remove])
        return attribute_names