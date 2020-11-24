# Written by Lior Bass 1/4/2018
import logging
import time

from math import ceil

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from commons.commons import replace_nominal_class_to_numeric
from commons.data_frame_creator import DataFrameCreator
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd


class Data_Handler(AbstractController):
    def __init__(self, db, targeted_class_name):
        AbstractController.__init__(self, db)
        self._default_cols = ['author_guid', 'attribute_name', 'attribute_value', 'author_type']
        self.optional_classes = ['bad_actor', 'good_actor']
        self._fill_empty = self._config_parser.eval(self.__class__.__name__, "fill_empty")
        self._to_replace_to_numerals = self._config_parser.eval(self.__class__.__name__, "to_replace_to_numerals")
        self._to_replace_authors_guid_author_name_to_authors_guid = False
        self._target_class_name = targeted_class_name
        self._author_guid= self._config_parser.eval(self.__class__.__name__, "id_field")
        self._remove_features_by_prefix = self._config_parser.eval(self.__class__.__name__, "remove_features_by_prefix")
        self._select_features_by_prefix = self._config_parser.eval(self.__class__.__name__, "select_features_by_prefix")

    def _initialize_dataframe(self, is_labeled):
        self._replace_authors_guid_author_name_to_authors_guid()
        author_features_objects = self._db.get_author_features_by_author_id_field(self._author_guid,
                                                                                  self._target_class_name, is_labeled)
        self._check_is_empty_table(author_features_objects)
        author_features_data_frame = self._convert_labeled_only_authors_features_to_dataframe(author_features_objects)
        return author_features_data_frame

    def _create_labeled_author_author_features_data_frame(self):
        author_features_data_frame = self._initialize_dataframe(is_labeled=True)
        author_features_data_frame, authors_label = self._match_author_label_to_author_features(
            author_features_data_frame)
        return author_features_data_frame, authors_label

    def _check_is_empty_table(self, author_features_objects):
        if not author_features_objects:
            logging.info("The table AUTHOR_FEATURES has no records. Execution is stopped, bye...")
            exit(0)

    def _remove_and_select_feature_from_dataframe(self, dataframe, feature_to_remove, feature_to_select):
        if len(feature_to_remove) > 0:
            dataframe = self._remove_features(feature_to_remove, dataframe)
        if len(feature_to_select) > 0:
            dataframe = dataframe[feature_to_select]
        return dataframe

    def _remove_features(self, features_to_remove, dataframe):
        dataframe_columns = dataframe.columns
        for unnecessary_feature in features_to_remove:
            if unnecessary_feature in dataframe_columns:
                dataframe.pop(unnecessary_feature)

        return dataframe

    def _get_author_features_dataframe(self):
        start_time = time.time()
        logging.info(
            "_get_author_features_dataframe started for " + self.__class__.__name__ + " started at " + str(start_time))
        data_frame_creator = DataFrameCreator(self._db)
        data_frame_creator.create_author_features_data_frame()
        author_features_dataframe = data_frame_creator.get_author_features_data_frame()

        end_time = time.time()
        logging.info(
            "_get_author_features_dataframe ended for " + self.__class__.__name__ + " ended at " + str(end_time))
        return author_features_dataframe

    def _replace_missing_values(self, author_features_data_frame):
        dataframe = author_features_data_frame
        dataframe.dropna(axis=1, how='all', inplace=True)
        if self._fill_empty == 'zero':
            dataframe = author_features_data_frame.fillna(0).dropna(axis=1, how='all')
        elif self._fill_empty == 'mean':
            dataframe = dataframe.fillna(dataframe.mean()).dropna(axis=1, how='all')
        return dataframe

    def _match_author_label_to_author_features(self, author_features_dataframe):
        author_features_data_frame = self._pivot_dataframe(author_features_dataframe)
        if self._target_class_name not in author_features_data_frame:
            authors_label = self._get_author_label_dataframe(author_features_dataframe)
            author_features_data_frame = author_features_data_frame.reset_index()
            author_features_data_frame = pd.merge(author_features_data_frame, authors_label, left_on=self._author_guid,
                                                  right_on=self._author_guid)
        authors_label = author_features_data_frame[self._target_class_name]
        author_features_data_frame.drop(self._target_class_name, axis=1, inplace=True)
        return author_features_data_frame, authors_label

    def _get_author_label_dataframe(self, author_features_dataframe):
        authors_label_dataframe = author_features_dataframe[[self._author_guid, self._target_class_name]]
        authors_label = authors_label_dataframe.drop_duplicates(subset=self._author_guid)
        authors_label.set_index(self._author_guid)
        return authors_label

    def _convert_labeled_only_authors_features_to_dataframe(self, author_features):
        logging.info("Start converting authors features into a dataframe")
        indices = [0, 3, 4, 5]
        data_frame = pd.DataFrame([[i[j] for j in indices] for i in author_features], columns=self._default_cols)
        logging.info("Finished converting authors features into a dataframe")
        return data_frame

    def _pivot_dataframe(self, df):
        logging.info("Start pivoting authors features rows into columns")
        df.drop_duplicates(['author_guid','attribute_name'],keep='first',inplace=True)
        pivoted = df.pivot(index = self._author_guid, columns='attribute_name', values='attribute_value')
        logging.info("Finished pivoting authors features rows into columns")
        return pivoted

    def _replace_authors_guid_author_name_to_authors_guid(self):
        if self._to_replace_authors_guid_author_name_to_authors_guid:
            self._db.convert_auhtor_feature_author_id_form_author_name_to_author_guid(self._db)

    def _replace_to_numerals(self, authors_labels, optional_classes):
        for label in list(optional_classes.keys()):
            authors_labels = authors_labels.replace(to_replace=label, value=optional_classes[label])
        return authors_labels
    def _convert_columns_to_special_int(self, df):
        for column in df:
            try:
                df[column].apply(pd.to_numeric)
            except:
                logging.info("replacing numerals in column "+ str(column))
                transformed = self._convert_column_with_strings_to_type_ints(df[column])
                df[column] = transformed
        return df.apply(pd.to_numeric)

    def _convert_column_with_strings_to_type_ints(self, column):
        le = LabelEncoder()
        le.fit(column.unique())
        transformed = le.transform(column)
        return transformed

    def _remove_and_select_features_by_prefix(self, authors_dataframe):
        if len(self._remove_features_by_prefix)>0:
            regex= r''
            for prefix in self._remove_features_by_prefix:
                regex += r'|^'+prefix
            regex =regex[1:]
            authors_dataframe = authors_dataframe[authors_dataframe.columns.drop(list(authors_dataframe.filter(regex=regex)))]
        if len(self._select_features_by_prefix)>0:
            regex=r''
            for prefix in self._select_features_by_prefix:
                authors_dataframe = authors_dataframe.select(lambda col: col.startswith(prefix), axis=1)
        return authors_dataframe

    def get_the_k_fragment_from_dataset(self, dataframe, labels, required_fragment, k):
        fragment_size = int(ceil(len(dataframe) / float(k)))
        first_index = (required_fragment) * fragment_size
        if required_fragment==k-1:
            last_index = len(dataframe)
        else:
            last_index = first_index + fragment_size
        test_set = dataframe[first_index:last_index]
        train_set = dataframe.drop(dataframe.index[list(range(first_index, last_index))])
        test_labels = labels[first_index:last_index]
        train_labels = labels.drop(labels.index[first_index:last_index])
        return test_set, train_set, test_labels, train_labels

    def get_labeled_authors_feature_dataframe_for_classification(self, removed_features,
                                                                 selected_features, optional_classes):
        authors_dataframe, authors_labels = self._create_labeled_author_author_features_data_frame()
        dataframe = self._remove_and_select_features_by_prefix(authors_dataframe)
        logging.info("Converting to numerals and replacing nulls")
        try:
            del dataframe[self._author_guid]
            dataframe = dataframe.apply(pd.to_numeric)  # convert all chars to numerics
            if self._to_replace_to_numerals:
                authors_labels = self._replace_to_numerals(authors_labels, optional_classes)
        except Exception as e:
            dataframe = self._convert_columns_to_special_int(dataframe)
            if self._to_replace_to_numerals:
                authors_labels = self._replace_to_numerals(authors_labels, optional_classes)
        dataframe = self._replace_missing_values(dataframe)
        return dataframe, authors_labels

    def get_unlabeled_authors_feature_dataframe_for_classification(self, replace_missing_values, removed_features,
                                                                   selected_features):
        author_features_data_frame = self._initialize_dataframe(is_labeled=False)
        author_features_data_frame = self._pivot_dataframe(author_features_data_frame)
        author_features_data_frame = self._remove_and_select_feature_from_dataframe(author_features_data_frame,
                                                                                    removed_features, selected_features)
        author_features_data_frame = self._replace_missing_values(author_features_data_frame)
        author_features_data_frame = author_features_data_frame.apply(pd.to_numeric)

        return author_features_data_frame


