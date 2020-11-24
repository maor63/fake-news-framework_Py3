# Created by shay at 27/09/2017
from .base_feature_generator import BaseFeatureGenerator
from sklearn.feature_extraction import FeatureHasher
from commons.method_executor import Method_Executor


class Image_Tags_Feature_Generator(Method_Executor):
    def __init__(self, db, **kwargs):
        Method_Executor.__init__(self, db)
        self._post_id_author_guid_dict = self._db.get_post_id_to_author_guid_mapping()
        self._word_id_dict = {}
        self._id_word_dict = {}
        self._author_features = []
        self._post_id_words_dict = {}
        self._authors_image_tags = self._db.get_authors_and_image_tags()
        self._word_id = 1



    '''writes the tags to the DB in an ordinal manner, giving each tag an ID and write it to the DB.
    the attribute_name field is in the format "image_tags_<tag>" and the value is the ID of the tag '''
    def ordinal(self):
        self._create_word_id_and_id_word_dict()
        post_id_word_ids_dict = {}
        post_ids = list(self._post_id_words_dict.keys())
        for post_id in post_ids:
            words = self._post_id_words_dict[post_id]
            words_ids = self._convert_tags_to_tag_ids(words)
            post_id_word_ids_dict[post_id] = words_ids
            author_guid = self._post_id_author_guid_dict[post_id]
            for word_id in words_ids:
                attribute_name = 'image_tags_' + self._id_word_dict[word_id].replace(" ", "_")
                author_feature = self._db.create_author_feature(author_guid, str(attribute_name),
                                                                word_id)
                self._author_features.append(author_feature)

        self._db.add_author_features(self._author_features)

    '''write the tags to the DB with dummy features, for each author writes a list of all the words and if 
    they exist or not the attribute name field is in the format "image_tags_is_contains_word_<tag>" and the 
    value is either 0 or 1'''
    def dummy(self):
        self._create_word_id_and_id_word_dict()
        post_id_tags_exists_dict = {}
        post_ids = list(self._post_id_words_dict.keys())
        for post_id in post_ids:
            tags = self._post_id_words_dict[post_id]
            tag_is_exist_dict = self._create_tag_is_exist_dict(tags)
            post_id_tags_exists_dict[post_id] = tag_is_exist_dict
            author_guid = self._post_id_words_dict[post_id]
            features_dict = post_id_tags_exists_dict[post_id]
            for feature in list(features_dict.keys()):
                attribute_name = 'image_tags_is_contains_word_' + feature.replace(" ", "_")
                value = features_dict[feature]
                author_feature = self._db.create_author_feature(author_guid, str(attribute_name),
                                                                value)
                self._author_features.append(author_feature)
        self._db.add_author_features(self._author_features)




    def _create_word_id_and_id_word_dict(self):
        for author_and_image_tag in self._authors_image_tags:
            post_id = author_and_image_tag[0]
            tags = author_and_image_tag[2]
            words = tags.split(',')
            self._post_id_words_dict[post_id] = words
            for word in words:
                if word not in list(self._word_id_dict.keys()):
                    self._word_id_dict[word] = self._word_id
                    self._id_word_dict[self._word_id] = word
                    self._word_id += 1

    def _create_tag_is_exist_dict(self, author_tags):
        tag_is_exist_dict = dict()
        for tag in list(self._word_to_id_dict.keys()):
            if tag in author_tags:
                tag_is_exist_dict[tag] = 1
            else:
                tag_is_exist_dict[tag] = 0
        return tag_is_exist_dict

    def _convert_tags_to_tag_ids(self, author_tags):
        tag_ids = []
        for tag in author_tags:
            tag_id = self._word_id_dict[tag]
            tag_ids.append(tag_id)
        return tag_ids
