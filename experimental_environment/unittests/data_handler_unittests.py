#Written by Lior Bass 2/28/2018
import calendar
import logging
import time
import uuid
from unittest import TestCase

from DB.schema_definition import DB, AuthorFeatures, date
from DB.schema_definition import Author
from experimental_environment.refactored_experimental_enviorment.data_handler import Data_Handler


class Data_Handler_Unittests(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._data_handler = Data_Handler(self._db, 'author_type')
        self._authors_to_author_features_dict = {}

        self._fill_empty= True
        self._remove_features = []
        self._select_features = []
        self._label_text_to_value = {'good':0,'bad':1}

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    def test_basic_case(self):
        self._create_author_with_features('1','good',(10,11,12,13,14,15,16))
        self._create_author_with_features('2','bad', (20,21,22,23,24,25,26))
        self._create_author_with_features('3','good', (30,31,32,33,34,35,36))
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(self._remove_features, self._select_features, self._label_text_to_value)

        self.assertEqual(1,1)

    def test_remove_by_prefix(self):
        self._create_author('123','bad')
        self._create_author_feature_with_name('123',3,'feature_test')
        self._create_author_feature_with_name('123', 5, 'feature_test2')
        self._create_author_feature_with_name('123', 5, 'bla_bla')
        self._create_author_feature_with_name('123', 5, 'dada')
        self._create_author_feature_with_name('123', 6, 'bla_bli')
        self._data_handler._remove_features_by_prefix = ['feature_test','bla']
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(self._remove_features, self._select_features, self._label_text_to_value)
        feature_num = len(authors_features_dataframe.columns)
        self.assertEqual(feature_num, 1)


    def test_remove_by_prefix_2(self):
        self._create_author('123','bad')
        self._create_author_feature_with_name('123',3,'feature_test')
        self._create_author_feature_with_name('123', 5, 'feature_test2')
        self._create_author_feature_with_name('123', 5, 'bla_bla')
        self._create_author_feature_with_name('123', 5, 'dada')
        self._create_author_feature_with_name('123', 6, 'bla_bli')
        self._data_handler._remove_features_by_prefix = ['feature_test']
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(
            self._remove_features, self._select_features, self._label_text_to_value)
        feature_num = len(authors_features_dataframe.columns)
        self.assertEqual(3, feature_num)

    def test_select_by_prefix(self):
        self._create_author('123','bad')
        self._create_author_feature_with_name('123',3,'feature_test')
        self._create_author_feature_with_name('123', 5, 'feature_test2')
        self._create_author_feature_with_name('123', 5, 'bla_bla')
        self._create_author_feature_with_name('123', 5, 'dada')
        self._create_author_feature_with_name('123', 6, 'bla_bli')
        self._data_handler._select_features_by_prefix = ['feature_test']
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(
            self._remove_features, self._select_features, self._label_text_to_value)
        feature_num = len(authors_features_dataframe.columns)
        self.assertEqual(2, feature_num)

    def test_select_by_prefix2(self):
        self._create_author('123','bad')
        self._create_author_feature_with_name('123',3,'feature_test')
        self._create_author_feature_with_name('123', 5, 'feature_test2')
        self._create_author_feature_with_name('123', 5, 'bla_bla')
        self._create_author_feature_with_name('123', 5, 'bloom_bla')
        self._create_author_feature_with_name('123', 5, 'blada')
        self._create_author_feature_with_name('123', 6, 'bla_bli')
        self._data_handler._select_features_by_prefix = ['bla']
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(
            self._remove_features, self._select_features, self._label_text_to_value)
        feature_num = len(authors_features_dataframe.columns)
        self.assertEqual(3, feature_num)

    def test_fill_and_drop_nan(self):
        self._create_author_with_features('1','good',(10,None,12,None))
        self._create_author_with_features('2', 'bad', (20, 24, 22,None))
        self._create_author_with_features('3', 'bad', (30, 34, 32,None))
        self._data_handler._fill_empty = 'zero'
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(self._remove_features, self._select_features, self._label_text_to_value)
        null_val = authors_features_dataframe.iloc[0]['1']
        self.assertEqual(null_val,0)
        did_remove_empty_column = '3' not in authors_features_dataframe.columns
        self.assertTrue(did_remove_empty_column)
        self._data_handler._fill_empty= 'mean'
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(self._remove_features, self._select_features, self._label_text_to_value)
        null_val = authors_features_dataframe.iloc[0]['1']
        self.assertEqual(null_val,(24+34)/2)

    def test_get_split(self):
        self._auto_create_authors(4,7)
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(
            self._remove_features, self._select_features, self._label_text_to_value)
        test_set, train_set, test_labels, train_labels = self._data_handler.get_the_k_fragment_from_dataset(authors_features_dataframe,
                                                                                                            authors_labels,
                                                                    0, 2)
        self.assertEqual(test_set.iloc[0]['0'],11)
        self.assertEqual(test_set.iloc[1]['0'],21)
        test_set, train_set, test_labels, train_labels = self._data_handler.get_the_k_fragment_from_dataset(
            authors_features_dataframe,
            authors_labels,
            1, 2)
        self.assertEqual(test_set.iloc[0]['0'],31)
        self.assertEqual(test_set.iloc[1]['0'],41)

    def test_train_and_test_differ(self):
        author_number = 7
        self._auto_create_authors(author_number,9)
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(
            self._remove_features, self._select_features, self._label_text_to_value)
        test_set, train_set, test_labels, train_labels = self._data_handler.get_the_k_fragment_from_dataset(authors_features_dataframe,
                                                                                                            authors_labels,
                                                                                                            0, 7)
        for num in range(author_number):
            author_guid =(num+1)*10+1
            is_in_both = self._is_val_in_datatframe(test_set,author_guid)==self._is_val_in_datatframe(train_set,author_guid)
            if is_in_both:
                logging.info("in both " + str(author_guid))
            self.assertFalse(is_in_both)

        test_set, train_set, test_labels, train_labels = self._data_handler.get_the_k_fragment_from_dataset(authors_features_dataframe,
                                                                                                            authors_labels,
                                                                                                            6, 7)
        for num in range(author_number):
            author_guid =(num+1)*10+1
            is_in_both = self._is_val_in_datatframe(test_set,author_guid)==self._is_val_in_datatframe(train_set,author_guid)
            if is_in_both:
                logging.info("in both "+str(author_guid))
            self.assertFalse(is_in_both)

    def _auto_create_authors(self, author_num, num_of_features):
        for num in range(author_num):
            author_name = num+1
            feature = []
            for feature_name in range(num_of_features):
                feature.append(str(author_name*10+feature_name+1))
            author_type = str(author_name*1000+author_name)
            self._create_author_with_features(str(author_name),author_type, feature)

    def _compare_authors_features_to_author(self):
        pass
    
    def _create_author_with_features(self, author_guid, author_type, feature_values):
        self._create_author(author_guid,author_type)
        for feature_value in feature_values:
            self._create_author_feature(author_guid,feature_value)
        self._db.session.commit()

    def _create_author(self, guid, author_type):
        author = Author()
        author.name = str(guid)
        author.domain = 'Microblog'
        author.author_guid = str(guid)
        author.author_screen_name = 'TestUser1'
        author.author_type = author_type
        author.domain = 'Restaurant'
        author.author_osn_id = 1

        self._authors_to_author_features_dict[author.author_guid]=[]
        self._db.add_author(author)

    def _create_author_feature(self, author_guid, value):
        feature_name = str(len(self._authors_to_author_features_dict[author_guid]))
        self._create_author_feature_with_name(author_guid, value, feature_name)

    def _create_author_feature_with_name(self, author_guid, value, feature_name):
        author_feature = AuthorFeatures()
        author_feature.author_guid = author_guid
        author_feature.window_start = date('2010-01-01 00:00:00')
        author_feature.window_end = date('2020-01-01 23:59:59')
        author_feature.attribute_name = feature_name
        author_feature.attribute_value=value
        self._authors_to_author_features_dict[author_guid].append(author_feature)
        self._db.update_author_features((author_feature))
        self._db.session.commit()

    def _is_val_in_datatframe(self, df, value):
        for row in range(df.shape[0]):  # df is the DataFrame
            for col in range(df.shape[1]):
                if df.iloc[row][col] == value:
                    return True
        return False
    def _get_random_guid(self):
        return str(uuid.uuid4())
