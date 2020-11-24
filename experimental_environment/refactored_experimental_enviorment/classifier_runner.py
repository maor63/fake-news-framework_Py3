#Written by Lior Bass 1/11/2018
import pickle
import csv
import logging

from numpy import select
from pandas import DataFrame
from sklearn.externals import joblib

from experimental_environment.refactored_experimental_enviorment.data_handler import Data_Handler
from preprocessing_tools.abstract_controller import AbstractController


class Classifier_Runner(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self,db)
        self._target_field = self._config_parser.eval(self.__class__.__name__, "target_field")
        self._data_handler = Data_Handler(db, targeted_class_name=self._target_field)
        self._classifier_file_path_and_name = self._config_parser.eval(self.__class__.__name__, "classifier_file_path_and_name")
        self._selected_feature_file_path_and_name = self._config_parser.eval(self.__class__.__name__, "selected_feature_file_path_and_name")
        self._saved_prediction_file_path_and_name = self._config_parser.eval(self.__class__.__name__, "saved_prediction_file_path_and_name")

        self._fill_empty = self._config_parser.eval(self.__class__.__name__, "fill_empty")
        self._remove_features = self._config_parser.eval(self.__class__.__name__, "remove_features")
        self._select_features = self._config_parser.eval(self.__class__.__name__, "select_features")
        self._label_text_to_value = self._config_parser.eval(self.__class__.__name__, "label_text_to_value")

        self._classifier = self._load_file(self._classifier_file_path_and_name)

    def predict_on_all_unlabled(self):
        dataframe = self._data_handler.get_unlabeled_authors_feature_dataframe_for_classification(self._fill_empty, self._remove_features, self._select_features )
        features_names = self._load_file(self._selected_feature_file_path_and_name)
        dataframe = dataframe[features_names]
        dataframe= self._data_handler._replace_missing_values(dataframe)
        labels = self._classifier.predict(dataframe)
        self.save_prediction_to_file(dataframe,labels, self._label_text_to_value)

    def save_prediction_to_file(self, dataframe, labels, optinal_classes):
        inv_dict = self._invert_dict(optinal_classes)
        index_field = dataframe.index
        tuples = [(index_field[i], labels[i],inv_dict[labels[i]]) for i in range(len(labels))]
        try:
            with open(self._saved_prediction_file_path_and_name, 'wb') as csv_file:
                wr = csv.writer(csv_file, delimiter=',')
                wr.writerow(('index_field_value','label', 'textual_label'))
                for cdr in tuples:
                    row = cdr
                    wr.writerow(row)
        except Exception as e:
            logging.info("could not open the prediction file, additional info: "+e)

    def _load_file(self, file_name):
        try:
            with open(file_name, 'rb') as f:
                clf = joblib.load(f)
                return clf
        except Exception as e:
            logging.info("could not open the file:"+file_name+"\n additional info: "+e)

    def _invert_dict(self, dict):
        try:
            inv_dict = {v: k for k, v in dict.items()}
            return inv_dict
        except Exception as e:
            logging.info("verbal to numeral classification is not injective")