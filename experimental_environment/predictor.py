# Created by Aviad Elyashar at 12/07/2017
from sklearn.externals import joblib

from commons.data_frame_creator import DataFrameCreator
from preprocessing_tools.abstract_controller import AbstractController
from .dataframe_manipulator import Dataframe_Manipulator
import pandas as pd
import numpy as np


class Predictor(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

        self._trained_classifier_file_name = self._config_parser.eval(self.__class__.__name__, "trained_classifier_file_name")
        self._best_features_file_name = self._config_parser.eval(self.__class__.__name__, "best_features_file_name")
        self._full_path_trained_classifier_directory = self._config_parser.eval(self.__class__.__name__, "full_path_trained_classifier_directory")
        self._targeted_class_field_names = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_names")
        self._replace_missing_values = self._config_parser.eval(self.__class__.__name__, "replace_missing_values")
        self._indentifier_field_name = self._config_parser.eval(self.__class__.__name__, "indentifier_field_name")
        self._feature_names_to_remove = self._config_parser.eval(self.__class__.__name__, "feature_names_to_remove")
        self._selected_features = self._config_parser.eval(self.__class__.__name__, "selected_features")
        self._classifier_type_name = self._config_parser.eval(self.__class__.__name__, "classifier_type_name")
        self._num_of_features = self._config_parser.eval(self.__class__.__name__, "num_of_features")
        self._targeted_class_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_dict")
        self._path = self._config_parser.eval(self.__class__.__name__, "path")

    def execute(self, window_start=None):
        data_frame_creator = DataFrameCreator(self._db)
        data_frame_creator.create_author_features_data_frame()
        author_features_dataframe = data_frame_creator.get_author_features_data_frame()

        for targeted_class_field_name in self._targeted_class_field_names:
            labeled_features_dataframe = author_features_dataframe.loc[author_features_dataframe[targeted_class_field_name].notnull()]
            unlabeled_features_dataframe = author_features_dataframe.loc[author_features_dataframe[targeted_class_field_name].isnull()]


            self._dataframe_manipulator = Dataframe_Manipulator(self._replace_missing_values, self._indentifier_field_name,
                                                targeted_class_field_name, self._feature_names_to_remove, self._selected_features)



            labeled_features_dataframe, labeled_targeted_class_series, labeled_indentifier_series = self._dataframe_manipulator.prepare_dataframe_for_learning(labeled_features_dataframe)
            unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_indentifier_series = self._dataframe_manipulator.prepare_dataframe_for_learning(unlabeled_features_dataframe)

            best_feature_names = self.get_best_feature_names()
            original_feature_names = list(labeled_features_dataframe.columns.values)

            features_to_remove = self._dataframe_manipulator.calculate_features_to_remove(best_feature_names, original_feature_names)

            labeled_features_dataframe = self._dataframe_manipulator.remove_features(features_to_remove, labeled_features_dataframe)
            unlabeled_features_dataframe = self._dataframe_manipulator.remove_features(features_to_remove, unlabeled_features_dataframe)

            unlabeled_features_dataframe = self._verify_number_of_columns_for_dataframe(best_feature_names, unlabeled_features_dataframe)

            best_classifier = self.get_trained_classifier()

            predictions_series, predictions_proba_series = self._predict_classifier(best_classifier,
                                                                                    unlabeled_features_dataframe)

            self._write_predictions_into_file(self._classifier_type_name, self._num_of_features,
                                              unlabeled_indentifier_series, predictions_series,
                                              predictions_proba_series)



    def get_trained_classifier(self):
        full_path_selected_model = self._full_path_trained_classifier_directory + self._trained_classifier_file_name

        trained_classifier = joblib.load(full_path_selected_model)
        return trained_classifier

    def get_best_feature_names(self):
        full_path_selected_features = self._full_path_trained_classifier_directory + self._best_features_file_name

        selected_features = joblib.load(full_path_selected_features)
        return selected_features


    def _predict_classifier(self, selected_classifier, unlabeled_author_dataframe):
        predictions = selected_classifier.predict(unlabeled_author_dataframe)
        predictions_series = pd.Series(predictions)

        predictions_series = self._replace_predictions_class_from_int_to_string(predictions_series)

        predictions_proba = selected_classifier.predict_proba(unlabeled_author_dataframe)

        optional_classes = list(self._targeted_class_dict.keys())
        num_of_classes = len(optional_classes)
        if num_of_classes == 2:
            predictions_proba_series = pd.Series(predictions_proba[:, 1])
            #max_predictions_proba = np.array(predictions_proba).max(axis=1)
            #predictions_proba_series = pd.Series(max_predictions_proba)
        elif num_of_classes > 2:
            predictions_proba_ndarray = np.array(predictions_proba)
            max_predictions_proba = predictions_proba_ndarray.max(axis=1)
            predictions_proba_series = pd.Series(max_predictions_proba)
        return predictions_series, predictions_proba_series


    def _write_predictions_into_file(self, classifier_type_name, num_of_features,
                                     unlabeled_index_field_series, predictions_series, predictions_proba_series):

        for targeted_class_field_name in self._targeted_class_field_names:
            unlabeled_dataframe_with_prediction = pd.DataFrame(unlabeled_index_field_series,
                                                               columns=[self._indentifier_field_name])

            unlabeled_dataframe_with_prediction.reset_index(drop=True, inplace=True)
            unlabeled_dataframe_with_prediction["predicted"] = predictions_series
            unlabeled_dataframe_with_prediction["prediction"] = predictions_proba_series

            full_path = self._path + "predictions_on_unlabeled_authors_" + targeted_class_field_name + "_" + \
                        classifier_type_name + "_" + str(num_of_features) + "_features.csv"
            # results_dataframe.to_csv(full_path)
            unlabeled_dataframe_with_prediction.to_csv(full_path, index=False)

            table_name = targeted_class_field_name + "_unlabeled_predictions"
            self._db.drop_unlabeled_predictions(table_name)

            engine = self._db.engine
            unlabeled_dataframe_with_prediction.to_sql(name=table_name, con=engine)
            unlabeled_dataframe_with_prediction_json = unlabeled_dataframe_with_prediction.to_json(orient='records')
            x = 3

    def _replace_predictions_class_from_int_to_string(self, predictions_series):
        predictions_series = self._replace_numeric_class_to_nominal(predictions_series)
        return predictions_series

    def _replace_numeric_class_to_nominal(self, dataframe):
        for targeted_class_name, num in self._targeted_class_dict.items():
            dataframe = dataframe.replace(to_replace=num, value=targeted_class_name)
        return dataframe

    def _verify_number_of_columns_for_dataframe(self, best_feature_names, unlabeled_features_dataframe):
        num_of_best_features = len(best_feature_names)
        unlabeled_features = list(unlabeled_features_dataframe.columns.values)
        num_of_unlabeled_features = len(unlabeled_features)
        if num_of_unlabeled_features < num_of_best_features:
            features_to_add_to_existing_dataframe = self._dataframe_manipulator.calculate_features_to_remove(unlabeled_features, best_feature_names)

            unlabeled_features_dataframe = pd.concat([unlabeled_features_dataframe, pd.DataFrame(columns=list(features_to_add_to_existing_dataframe))])
            unlabeled_features_dataframe = unlabeled_features_dataframe.fillna(0)
        return unlabeled_features_dataframe


