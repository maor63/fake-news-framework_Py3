# Created by Aviad at 10/04/2016

import os
import random
import re
import copy
from itertools import combinations
import csv
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import pickle
import matplotlib.pyplot as plt
from nltk.metrics.scores import f_measure
from sympy.stats.rv import probability

# import xgboost as xgb
from sklearn import tree, svm
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import joblib

from commons.commons import *
from commons.method_executor import Method_Executor
from commons.data_frame_creator import DataFrameCreator
from commons.consts import PerformanceMeasures, Classifiers
import random


class ExperimentalEnvironment(Method_Executor):

    def __init__(self, db):
        Method_Executor.__init__(self, db)

        self._performance_measures = [PerformanceMeasures.AUC,
                                      PerformanceMeasures.ACCURACY,
                                      PerformanceMeasures.PRECISION,
                                      PerformanceMeasures.RECALL,
                                      PerformanceMeasures.CONFUSION_MATRIX,
                                      PerformanceMeasures.SELECTED_FEATURES,
                                      PerformanceMeasures.CORRECTLY_CLASSIFIED,
                                      PerformanceMeasures.INCORRECTLY_CLASSIFIED,
                                      PerformanceMeasures.LRAP,
                                      PerformanceMeasures.LRL,
                                      PerformanceMeasures.spearman_correlation,
                                      PerformanceMeasures.spearman_coefficient_p_value,
                                      ]

        self._divide_lableled_by_percent_training_size = self._config_parser.eval(self.__class__.__name__,
                                                                                  "divide_lableled_by_percent_training_size")
        self._k_for_fold = self._config_parser.eval(self.__class__.__name__, "k_for_fold")
        self._classifier_type_names = self._config_parser.eval(self.__class__.__name__, "classifier_type_names")
        self._selected_features = self._config_parser.eval(self.__class__.__name__, "selected_features")

        self._removed_features = self._config_parser.eval(self.__class__.__name__, "removed_features")
        self._optional_classes = self._config_parser.eval(self.__class__.__name__, "optional_classes")
        self._index_field = self._config_parser.get(self.__class__.__name__, "index_field")

        self._results_file_name = self._config_parser.get(self.__class__.__name__, "results_file_name")
        self._path = self._config_parser.get(self.__class__.__name__, "path")
        self._backup_path = self._config_parser.get(self.__class__.__name__, "backup_path")
        self._num_of_features_to_train = self._config_parser.eval(self.__class__.__name__, "num_of_features_to_train")
        self._full_path_model_directory = self._config_parser.get(self.__class__.__name__, "full_path_model_directory")
        self._prepared_classifier_name = self._config_parser.get(self.__class__.__name__, "prepared_classifier_name")
        # self._fill_predictions_to_unlabeled_authors = self.config_parser.eval(self.__class__.__name__, "fill_predictions_to_unlabeled_authors")
        self._column_names_for_results_table = self._config_parser.eval(self.__class__.__name__,
                                                                        "column_names_for_results_table")
        self._results_table_file_name = self._config_parser.get(self.__class__.__name__, "results_table_file_name")

        # self._prepared_classifier_name = self._config_parser.get(self.__class__.__name__, "prepared_classifier_name")
        self._targeted_class_name = self._config_parser.get(self.__class__.__name__, "targeted_class_name")
        # self._num_of_features = self._config_parser.eval(self.__class__.__name__, "num_of_features")
        self._classifier_type_names = self._config_parser.eval(self.__class__.__name__, "classifier_type_names")
        self._trained_classifier_type_name = self._config_parser.get(self.__class__.__name__,
                                                                     "trained_classifier_type_name")
        self._trained_classifier_num_of_features = self._config_parser.eval(self.__class__.__name__,
                                                                            "trained_classifier_num_of_features")

        # self._train_one_class_classifier_and_predict = self._config_parser.eval(self.__class__.__name__, "train_one_class_classifier_and_predict")
        # self._train_one_class_classifier_by_k_best_and_predict = self._config_parser.eval(self.__class__.__name__, "train_one_class_classifier_by_k_best_and_predict")

        self._replace_missing_values = self._config_parser.eval(self.__class__.__name__, "replace_missing_values")
        # self._transfer_learning = self._config_parser.eval(self.__class__.__name__, "transfer_learning")
        # self._transfer_algo = self._config_parser.eval(self.__class__.__name__, "transfer_algo")
        # self._num_neighbors = self._config_parser.eval(self.__class__.__name__, "num_neighbors")

        self._num_iterations = self._config_parser.eval(self.__class__.__name__, "num_iterations")
        self._num_of_iterations = self._config_parser.eval(self.__class__.__name__, "num_of_iterations")
        #  self._source_domains = self._config_parser.eval(self.__class__.__name__, "source_domains")
        #  self._source_input_type = self._config_parser.eval(self.__class__.__name__, "source_input_type")
        #  self._source_input_path = self._config_parser.eval(self.__class__.__name__, "source_input_path")
        #
        #  self._target_train_test_split = self._config_parser.eval(self.__class__.__name__, "target_train_test_split")
        #  self._target_train_percent_limit = self._config_parser.eval(self.__class__.__name__, "target_train_percent_limit")
        #  self._target_test_percent_limit = self._config_parser.eval(self.__class__.__name__, "target_test_percent_limit")
        #  self._transfer_instances = self._config_parser.eval(self.__class__.__name__, "transfer_instances")
        #
        #
        #  self._target_domains = self._config_parser.eval(self.__class__.__name__, "target_domains")
        #  self._target_input_type = self._config_parser.eval(self.__class__.__name__, "target_input_type")
        #  self._target_input_path = self._config_parser.eval(self.__class__.__name__, "target_input_path")
        #  self._feature_scaling = self._config_parser.eval(self.__class__.__name__, "feature_scaling")
        #  self._feature_selection = self._config_parser.eval(self.__class__.__name__, "feature_selection")
        #
        #  self._stdev_threshold = self._config_parser.eval(self.__class__.__name__, "stdev_threshold")

        # fake news promoters experiment
        self._replicate_authors_by = self._config_parser.eval(self.__class__.__name__, "replicate_authors_by")
        self._validation_results_table_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                            "validation_results_table_file_name")
        self._test_results_table_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                      "test_results_table_file_name")
        self._datasets_actors_distribution_results_table_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                                              "datasets_actors_distribution_results_table_file_name")

        # fake news promoters labeling during cross validation fields
        self._fake_and_real_news_labels_dict = self._config_parser.eval(self.__class__.__name__,
                                                                        "fake_and_real_news_labels_dict")
        self._targeted_class_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_dict")
        self._min_num_of_distinct_claims_author_involved = self._config_parser.eval(self.__class__.__name__,
                                                                                    "min_num_of_distinct_claims_author_involved")
        self._min_num_of_published_posts = self._config_parser.eval(self.__class__.__name__,
                                                                    "min_num_of_published_posts")
        self._threshold = self._config_parser.eval(self.__class__.__name__, "threshold")
        self._prediction_function = self._config_parser.eval(self.__class__.__name__, "prediction_function")
        self._target_class_classifier_dictionary = {}

    def set_up(self):
        pass

    # A. Take all the authors
    # B. Divide to training and test sets by percents.
    # C. Train on 60% precent: Duplicate each author according to the number of claims he was involved in.
    #    Find the best classifier.
    # D. Predict on the 40% percent for validation.
    # E. Save results

    def fake_and_real_news_promoters_experiment(self):
        self._dataset_authors_distribution_tuples = []
        author_features_dataframe = self._get_author_features_dataframe()
        labeled_authors_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                author_features_dataframe)

        self._update_dataset_statistics('All_dataset', labeled_authors_df)

        column_names = list(labeled_authors_df.columns.values)
        targeted_class_dfs = self._create_labeled_dfs(labeled_authors_df)

        validation_result_tuples = []
        test_set_result_tuples = []

        author_guid_binary_verdict_results_df = pd.read_csv(self._path + "author_guid_binary_verdict_results.csv")

        for training_size_percent in self._divide_lableled_by_percent_training_size:
            # for classifier_type_name in self._classifier_type_names:

            original_training_df = self._build_training_set(training_size_percent, targeted_class_dfs)
            self._update_dataset_statistics('original_training_set', original_training_df)

            original_training_df_indexes = original_training_df.index.tolist()

            test_df = labeled_authors_df[~labeled_authors_df.index.isin(original_training_df_indexes)]
            self._update_dataset_statistics('test_set', test_df)
            test_df, test_targeted_class_series, test_index_field_series = self._prepare_dataframe_for_learning(
                test_df)

            if self._replicate_authors_by is not None:
                replicated_training_df = self._replicate_authors_by_num_of_labeled_claim_posts(original_training_df,
                                                                                               author_guid_binary_verdict_results_df)
                training_df = replicated_training_df
            else:
                training_df = original_training_df

            self._update_dataset_statistics('replicated_validation_set_from_the_training_set_only', training_df)
            original_training_df_for_validation, training_targeted_class_series, training_index_field_series = self._prepare_dataframe_for_learning(
                training_df)

            for classifier_type_name in self._classifier_type_names:
                for num_of_features in self._num_of_features_to_train:

                    training_df, training_df_column_names = self._reduce_dimensions_by_num_of_features(
                        original_training_df_for_validation,
                        training_targeted_class_series,
                        num_of_features)

                    total_predictions = []
                    total_targeted_classes_for_predictions = []

                    k_folds, valid_k = self._select_valid_k(training_targeted_class_series)

                    i = 0
                    for train_indexes, validation_indexes in k_folds:
                        i += 1
                        msg = "\r K Fold cross validation for classifier {0} #features {1} {2}/{3}".format(
                            classifier_type_name, num_of_features, i, valid_k)
                        print(msg, end='')

                        train_set_dataframe, validation_set_dataframe, train_class, validation_class = \
                            self._create_train_and_test_dataframes_and_classes(training_df, train_indexes,
                                                                               validation_indexes,
                                                                               training_targeted_class_series)

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)
                        selected_classifier.fit(train_set_dataframe, train_class)

                        columns_to_remove = list(set(column_names) - set(training_df_column_names))
                        validation_set_dataframe = self._remove_features(columns_to_remove, validation_set_dataframe)

                        if classifier_type_name == "XGBoost":
                            validation_set_dataframe = self._set_dataframe_columns_types(validation_set_dataframe)

                        predictions = selected_classifier.predict(validation_set_dataframe)
                        total_predictions.extend(predictions)
                        total_targeted_classes_for_predictions.extend(validation_class)

                        # predictions_proba = selected_classifier.predict_proba(test_df)

                    try:
                        auc_score = roc_auc_score(total_targeted_classes_for_predictions, total_predictions)
                    except:
                        auc_score = -1
                    accuracy = accuracy_score(total_targeted_classes_for_predictions, total_predictions)
                    f1 = f1_score(total_targeted_classes_for_predictions, total_predictions)
                    precision = precision_score(total_targeted_classes_for_predictions, total_predictions)
                    recall = recall_score(total_targeted_classes_for_predictions, total_predictions)
                    # conf_matrix = confusion_matrix(validation_class, predictions)

                    result_tuple = (classifier_type_name, num_of_features, auc_score, accuracy, f1, precision,
                                    recall, training_df_column_names)
                    validation_result_tuples.append(result_tuple)

        validation_results_df = pd.DataFrame(validation_result_tuples,
                                             columns=['Classifier', '#Features', "AUC", "Accuracy", "F1",
                                                      "Precision", "Recall", "Selected Features"])
        validation_results_df.to_csv(self._path + self._validation_results_table_file_name, index=None)

        author_guid_max_auc = validation_results_df['AUC'].argmax()

        best_performance_df = validation_results_df.iloc[[author_guid_max_auc]]
        print("Max parrameters found in row: {0}".format(best_performance_df))

        best_classifier_name = best_performance_df['Classifier'].tolist()[0]
        best_performance_feature_names = best_performance_df['Selected Features'].tolist()[0]

        original_training_df, original_training_class_series, original_training_index_field_series = self._prepare_dataframe_for_learning(
            original_training_df)

        best_classifier = self._select_classifier_by_type(best_classifier_name)

        original_column_names = list(original_training_df.columns.values)
        features_to_remove = self._calculate_features_to_remove(best_performance_feature_names, original_column_names)
        reduced_original_training_df = self._remove_features(features_to_remove, original_training_df)

        best_classifier.fit(reduced_original_training_df, original_training_class_series)
        self._save_trained_model(best_classifier, best_classifier_name, len(best_performance_feature_names),
                                 best_performance_feature_names)

        test_df = self._remove_features(columns_to_remove, test_df)

        if best_classifier == "XGBoost":
            test_df = self._set_dataframe_columns_types(test_df)

        reduced_test_df = self._remove_features(features_to_remove, test_df)

        predictions = best_classifier.predict(reduced_test_df)
        # predictions_proba = selected_classifier.predict_proba(test_df)

        try:
            auc_score = roc_auc_score(test_targeted_class_series, predictions)
        except:
            auc_score = -1
        accuracy = accuracy_score(test_targeted_class_series, predictions)
        f1 = f1_score(test_targeted_class_series, predictions)
        precision = precision_score(test_targeted_class_series, predictions)
        recall = recall_score(test_targeted_class_series, predictions)
        # conf_matrix = confusion_matrix(validation_class, predictions)

        test_set_result_tuple = (
            best_classifier_name, len(best_performance_feature_names), auc_score, accuracy, f1, precision,
            recall, best_performance_feature_names)
        test_set_result_tuples.append(test_set_result_tuple)

        test_results_df = pd.DataFrame(test_set_result_tuples,
                                       columns=['Classifier', '#Features', "AUC",
                                                "Accuracy", "F1", "Precision", "Recall", "Selected Features"])
        test_results_df.to_csv(self._path + self._test_results_table_file_name, index=None)

        datasets_actors_distribution_results_df = pd.DataFrame(self._dataset_authors_distribution_tuples,
                                                               columns=['Dataset', self._optional_classes[0],
                                                                        self._optional_classes[1],
                                                                        'predicted_accuracy_by_majority_class'])
        datasets_actors_distribution_results_df.to_csv(
            self._path + self._datasets_actors_distribution_results_table_file_name, index=None)

    def assigning_fake_and_real_news_promoters_while_cross_validation_experiment(self):
        claim_post_author_connections = self._db.get_claim_post_author_connections_with_verdict()

        self._claim_post_author_connection_df = pd.DataFrame(claim_post_author_connections,
                                                             columns=['claim_id', 'post_id', 'author_guid', 'verdict'])

        self._convert_verdict_to_binary_problem(self._claim_post_author_connection_df)

        claims = self._db.get_claims()

        # claims = claims[0:15]

        self._claim_num_verdict_df = self._create_claim_verdict_df(claims)

        self._convert_verdict_to_binary_problem(self._claim_num_verdict_df)

        claim_verdict_series = self._claim_num_verdict_df['verdict']

        k_folds, valid_k = self._select_valid_k(claim_verdict_series)

        i = 0

        total_confidence_predictions = []
        total_binary_predictions = []
        total_test_class = []
        test_performance_results = []
        # will save the prediction for test claim in each iteration
        self._test_claim_prediction_tuples = []
        self._total_author_guid_statistics_df = pd.DataFrame()
        for train_indexes, test_indexes in k_folds:
            i += 1
            print("K fold cross validation iteration {0}/{1}".format(i, valid_k))
            training_df, test_df, training_class, test_class = \
                self._create_train_and_test_dataframes_and_classes(self._claim_num_verdict_df, train_indexes,
                                                                   test_indexes,
                                                                   claim_verdict_series)

            training_claim_post_author_connection_df = self._build_training_claim_post_author_connections(training_df)

            author_guid_statistics_df = self._label_authors_based_on_training_claims(
                training_claim_post_author_connection_df,
                i)

            confidence_predictions, binary_predictions = self._predict_test_claims(test_df, author_guid_statistics_df,
                                                                                   i)

            self._total_author_guid_statistics_df = self._total_author_guid_statistics_df.append(
                author_guid_statistics_df, ignore_index=True)
            total_confidence_predictions.extend(confidence_predictions)
            total_binary_predictions.extend(binary_predictions)
            total_test_class.extend(test_class)

        test_claim_predictions_df = pd.DataFrame(self._test_claim_prediction_tuples,
                                                 columns=['test_claim_id', 'k_fold_iteration_num',
                                                          'fake_news_promoter_count',
                                                          'real_news_promoter_count', 'is_random_guess',
                                                          'confidence_prediction',
                                                          'binary_prediction', 'ground_truth'])
        test_claim_predictions_df.to_csv(self._path + "test_claim_predictions.csv")

        self._total_author_guid_statistics_df.to_csv(self._path + "total_author_guid_statistics.csv")

        test_performance_result_tuple = self._calculate_performance(total_test_class, total_confidence_predictions,
                                                                    total_binary_predictions, self._prediction_function)
        test_performance_results.append(test_performance_result_tuple)

        results_df = pd.DataFrame(test_performance_results, columns=['Classifier', "AUC", "Accuracy", "F1",
                                                                     "Precision", "Recall"])
        results_df.to_csv(self._path + self._results_table_file_name, index=None)

    def divide_to_training_and_test_by_percent_random(self):
        print("Running divide_to_training_and_test_by_percent_random")
        author_features_dataframe = self._get_author_features_dataframe()
        labeled_authors_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                author_features_dataframe)

        targeted_class_dfs = self._create_labeled_dfs(labeled_authors_df)

        column_names = list(labeled_authors_df.columns.values)
        result_tuples = []
        for classifier_type_name in self._classifier_type_names:
            for num_of_features in self._num_of_features_to_train:
                for training_size_percent in self._divide_lableled_by_percent_training_size:
                    for i in range(self._num_of_iterations):

                        msg = "\r Classifier: {0}, #features: {1}, training size: {2}, iteration: {3}".format(
                            classifier_type_name, num_of_features, training_size_percent, i)
                        print(msg, end="")

                        training_df = self._build_training_set(training_size_percent, targeted_class_dfs)

                        training_df, training_targeted_class_series, training_index_field_series = self._prepare_dataframe_for_learning(
                            training_df)

                        training_df_indexes = training_df.index.tolist()

                        test_df = labeled_authors_df[~labeled_authors_df.index.isin(training_df_indexes)]

                        test_df, test_targeted_class_series, test_index_field_series = self._prepare_dataframe_for_learning(
                            test_df)

                        training_df, training_df_column_names = self._reduce_dimensions_by_num_of_features(training_df,
                                                                                                           training_targeted_class_series,
                                                                                                           num_of_features)

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)

                        selected_classifier.fit(training_df, training_targeted_class_series)

                        columns_to_remove = list(set(column_names) - set(training_df_column_names))
                        test_df = self._remove_features(columns_to_remove, test_df)

                        if classifier_type_name == "XGBoost":
                            test_df = self._set_dataframe_columns_types(test_df)

                        predictions = selected_classifier.predict(test_df)
                        # predictions_proba = selected_classifier.predict_proba(test_df)

                        try:
                            auc_score = roc_auc_score(test_targeted_class_series, predictions)
                        except:
                            auc_score = -1
                        accuracy = accuracy_score(test_targeted_class_series, predictions)
                        f1 = f1_score(test_targeted_class_series, predictions)
                        precision = precision_score(test_targeted_class_series, predictions)
                        recall = recall_score(test_targeted_class_series, predictions)
                        conf_matrix = confusion_matrix(test_targeted_class_series, predictions)

                        result_tuple = (
                            classifier_type_name, num_of_features, training_size_percent, i, auc_score, accuracy, f1,
                            precision, recall, conf_matrix, training_df_column_names)
                        result_tuples.append(result_tuple)

            df = pd.DataFrame(result_tuples,
                              columns=['Classifier', '#Features', "%Training Size", "#Iteration", "AUC", "Accuracy",
                                       "F1", "Precision", "Recall", "Confusion Matrix", "Selected Features"])
            df.to_csv(self._path + self._results_table_file_name, index=None)

    def perform_k_fold_cross_validation_and_predict(self):
        self._target_class_classifier_dictionary = self._create_target_class_classifier_dictionary()
        self._max_classifier_dictionary = self._init_max_classifier_dict()

        author_features_dataframe = self._get_author_features_dataframe()

        unlabeled_features_dataframe = self._retreive_unlabeled_authors_dataframe(author_features_dataframe)
        unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_index_field_series = \
            self._prepare_dataframe_for_learning(unlabeled_features_dataframe)

        labeled_features_dataframe = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                        author_features_dataframe)

        labeled_features_dataframe, targeted_class_series, index_field_series = \
            self._prepare_dataframe_for_learning(labeled_features_dataframe)

        self._current_classifier_performance_dict = {}

        for classifier_type_name in self._classifier_type_names:
            for num_of_features in self._num_of_features_to_train:
                targeted_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                    labeled_features_dataframe, targeted_class_series, num_of_features)

                self._set_selected_features_into_dictionary(classifier_type_name, num_of_features,
                                                            dataframe_column_names)
                print("classifier_type_name = " + classifier_type_name)
                selected_classifier = self._select_classifier_by_type(classifier_type_name)

                if selected_classifier is not None:

                    k_folds, valid_k = self._select_valid_k(targeted_class_series)

                    print("Valid k is: " + str(valid_k))
                    i = 0
                    for train_indexes, test_indexes in k_folds:
                        i += 1
                        print("i = " + str(i))

                        train_set_dataframe, test_set_dataframe, train_class, test_class = self._create_train_and_test_dataframes_and_classes(
                            targeted_dataframe,
                            train_indexes, test_indexes, targeted_class_series)

                        selected_classifier.fit(train_set_dataframe, train_class)

                        predictions = selected_classifier.predict(test_set_dataframe)
                        predictions_proba = selected_classifier.predict_proba(test_set_dataframe)

                        classes = selected_classifier.classes_

                        for performance_measure in self._performance_measures:
                            self._update_performance_measures(classifier_type_name, num_of_features,
                                                              predictions, predictions_proba, performance_measure,
                                                              test_class, classes)

                    self._calculate_average_for_performance_measures(classifier_type_name, valid_k, num_of_features)

        self._write_results_into_file()
        self._write_results_as_table()

        if not unlabeled_features_dataframe.empty:
            self._create_best_classifier_train_save_and_predict(labeled_features_dataframe, targeted_class_series,
                                                                unlabeled_features_dataframe,
                                                                unlabeled_index_field_series,
                                                                unlabeled_targeted_class_series)
        else:
            print("The dataset is not include unlabeled authors!! so that is it!!")

    def predict_on_prepared_clssifier(self):
        selected_classifier = self._get_trained_classifier()

        author_features_dataframe = self._get_author_features_dataframe()
        unlabeled_features_dataframe = self._retreive_unlabeled_authors_dataframe(author_features_dataframe)
        unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_index_field_series = \
            self._prepare_dataframe_for_learning(unlabeled_features_dataframe)

        # unlabeled_features_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
        #    unlabeled_features_dataframe, unlabeled_targeted_class_series, self._num_of_features)

        predictions_series, predictions_proba_series = self._predict_classifier(selected_classifier,
                                                                                unlabeled_features_dataframe)

        self._write_predictions_into_file(self._trained_classifier_type_name, self._trained_classifier_num_of_features,
                                          unlabeled_index_field_series, predictions_series,
                                          predictions_proba_series)

    def train_one_class_classifier_and_predict(self):
        self._one_class_column_names = ['Combination', '#Bad_Actors_Training_Set',
                                        '#Bad_Actors_Errors_Test_Set', 'STDEV_Bad_Actors_Errors_Test_Set',
                                        '#Bad_Actors_Corrected_Test_Set', 'STDEV_Bad_Actors_Corrected_Test_Set',
                                        '#Good_Actors_Errors_Test_Set', 'STDEV_Good_Actors_Errors_Test_Set',
                                        '#Good_Actors_Errors_Test_Set', 'STDEV_Good_Actors_Errors_Test_Set',
                                        '#Total_Test_Set'
                                        ]

        self._one_class_dict = self._create_one_class_dictionary()

        labeled_features_dataframe, unlabeled_features_dataframe, targeted_class_series, \
        unlabeled_targeted_class_series, unlabeled_index_field_series = self._create_labeled_and_unlabeled_based_on_author_features()

        # path = self._path + "labeled_features_dataframe.txt"
        # labeled_features_dataframe.to_csv(path, sep=',')

        feature_names = list(labeled_features_dataframe.columns.values)
        self._train_one_class_classifiers_for_each_combination(labeled_features_dataframe, targeted_class_series)

        one_class_result_dataframe = pd.DataFrame(self._one_class_dict, columns=self._one_class_column_names)

        full_path = self._path + "one_class_results.csv"
        # results_dataframe.to_csv(full_path)
        one_class_result_dataframe.to_csv(full_path, index=False)

        best_combination_elements = self._find_best_combination(one_class_result_dataframe)

        labeled_features_dataframe, unlabeled_features_dataframe = self._create_labeled_and_unlabeled_based_on_combination(
            best_combination_elements, feature_names,
            labeled_features_dataframe, unlabeled_features_dataframe)

        one_class_classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

        one_class_classifier.fit(labeled_features_dataframe)
        unlabeled_predictions = one_class_classifier.predict(unlabeled_features_dataframe)
        distances = one_class_classifier.decision_function(unlabeled_features_dataframe)

        self._write_predictions_into_file("one_class", str(len(best_combination_elements)),
                                          unlabeled_index_field_series,
                                          unlabeled_predictions, distances)

    def train_one_class_classifier_by_k_best_and_predict(self):
        self._one_class_column_names = ['Combination', 'Num_of_features', '#Bad_Actors_Training_Set',
                                        '#Errors_Bad_Actors_Test_Set',
                                        'STDEV_Bad_Actors_Errors_Test_Set', '#Corrected_Bad_Actors_Test_Set',
                                        'STDEV_Corrected_Bad_Actors_Test_Set',
                                        '#Errors_Good_Actors_Test_Set', 'STDEV_Good_Actors_Errors_Test_Set',
                                        '#Corrects_Good_Actors_Test_Set', 'STDEV_Good_Actors_Corrects_Test_Set',
                                        '#Errors_Test_Set', 'STDEV_Errors_Test_Set',
                                        '#Corrects_Test_Set', 'STDEV_Corrects_Test_Set',
                                        '#Total_Test_Set']

        self._one_class_dict = self._create_one_class_dictionary()

        labeled_features_dataframe, unlabeled_features_dataframe, unlabeled_targeted_class_series, \
        unlabeled_index_field_series = self._create_unlabeled_authors_dataframe_and_raw_labeled_authors_dataframe()

        good_actors_dataframe = labeled_features_dataframe.loc[
            labeled_features_dataframe[self._targeted_class_name] == 'good_actor']
        manually_bad_actors_dataframe = labeled_features_dataframe.loc[
            (labeled_features_dataframe[self._targeted_class_name] == 'bad_actor') & (
                labeled_features_dataframe['author_sub_type'].isnull())]
        isis_bad_actors_dataframe = labeled_features_dataframe.loc[
            (labeled_features_dataframe[self._targeted_class_name] == 'bad_actor') & (
                    labeled_features_dataframe['author_sub_type'] == 'ISIS_terrorist')]

        good_actors_dataframe, good_actors_targeted_class_series, good_actors_index_field_series = \
            self._prepare_dataframe_for_learning(good_actors_dataframe)

        isis_bad_actors_dataframe, isis_bad_actors_targeted_class_series, isis_bad_actors_index_field_series = \
            self._prepare_dataframe_for_learning(isis_bad_actors_dataframe)

        for num_of_features in self._num_of_features_to_train:
            reduced_isis_bad_actors_dataframe, selected_column_names = self._reduce_dimensions_by_num_of_features(
                isis_bad_actors_dataframe, isis_bad_actors_targeted_class_series, num_of_features)

            feature_names = list(labeled_features_dataframe.columns.values)
            best_combination_set = set(selected_column_names)
            feature_names_set = set(feature_names)
            features_to_remove_set = feature_names_set - best_combination_set
            features_to_remove = list(features_to_remove_set)

            reduced_good_actors_dataframe = self._remove_features(features_to_remove, good_actors_dataframe.copy())

            combination_name = "+".join(selected_column_names)
            self._one_class_dict['Combination'].append(combination_name)
            self._one_class_dict['Num_of_features'].append(num_of_features)

            isis_bad_actors_training_set_size_count = 0
            isis_bad_actors_test_set_size_count = 0
            good_actors_test_set_size_count = 0

            bad_actors_test_set_errors_count = 0
            bad_actors_test_set_errors = []
            bad_actors_test_set_corrects_count = 0
            bad_actors_test_set_corrects = []

            good_actors_test_set_errors_count = 0
            good_actors_test_set_errors = []
            good_actors_test_set_corrects_count = 0
            good_actors_test_set_corrects = []

            total_test_set_errors_count = 0
            total_test_set_errors = []

            total_test_set_corrects_count = 0
            total_test_set_corrects = []

            test_set_total = 0

            k_folds, valid_k = self._select_valid_k(isis_bad_actors_targeted_class_series)
            for train_indexes, test_indexes in k_folds:
                isis_bad_actors_train_set_dataframe, isis_bad_actors_test_set_dataframe, train_class, test_class = self._create_train_and_test_dataframes_and_classes(
                    reduced_isis_bad_actors_dataframe,
                    train_indexes, test_indexes,
                    isis_bad_actors_targeted_class_series)
                training_size = isis_bad_actors_train_set_dataframe.shape[0]
                isis_bad_actors_training_set_size_count += training_size

                isis_bad_actors_test_set_size = isis_bad_actors_test_set_dataframe.shape[0]
                isis_bad_actors_test_set_size_count += isis_bad_actors_test_set_size

                good_actors_test_set_size = reduced_good_actors_dataframe.shape[0]
                good_actors_test_set_size_count += good_actors_test_set_size

                one_class_classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

                one_class_classifier.fit(isis_bad_actors_train_set_dataframe)

                bad_actors_test_set_predictions = one_class_classifier.predict(isis_bad_actors_test_set_dataframe)
                # distances = one_class_classifier.decision_function(test_set_dataframe)
                num_error_bad_actors_test_set = bad_actors_test_set_predictions[
                    bad_actors_test_set_predictions == -1].size
                bad_actors_test_set_errors.append(num_error_bad_actors_test_set)
                bad_actors_test_set_errors_count += num_error_bad_actors_test_set
                test_set_total += num_error_bad_actors_test_set

                num_correct_bad_actors_test_set = bad_actors_test_set_predictions[
                    bad_actors_test_set_predictions == 1].size
                bad_actors_test_set_corrects.append(num_correct_bad_actors_test_set)
                bad_actors_test_set_corrects_count += num_correct_bad_actors_test_set
                test_set_total += num_correct_bad_actors_test_set

                good_actors_test_set_predictions = one_class_classifier.predict(reduced_good_actors_dataframe)
                # distances = one_class_classifier.decision_function(test_set_dataframe)
                num_error_good_actors_test_set = good_actors_test_set_predictions[
                    good_actors_test_set_predictions == 1].size
                good_actors_test_set_errors.append(num_error_good_actors_test_set)
                good_actors_test_set_errors_count += num_error_good_actors_test_set
                test_set_total += num_error_good_actors_test_set

                num_correct_good_actors_test_set = good_actors_test_set_predictions[
                    good_actors_test_set_predictions == -1].size
                good_actors_test_set_corrects.append(num_correct_good_actors_test_set)
                good_actors_test_set_corrects_count += num_correct_good_actors_test_set
                test_set_total += num_correct_good_actors_test_set

                total_test_errors = num_error_bad_actors_test_set + num_error_good_actors_test_set
                total_test_set_errors_count += total_test_errors
                total_test_set_errors.append(total_test_errors)

                total_test_corrects = num_correct_bad_actors_test_set + num_correct_good_actors_test_set
                total_test_set_corrects_count += total_test_corrects
                total_test_set_corrects.append(total_test_corrects)

            isis_bad_actors_training_set_size_count = float(isis_bad_actors_training_set_size_count) / self._k_for_fold
            self._one_class_dict['#Bad_Actors_Training_Set'].append(isis_bad_actors_training_set_size_count)

            bad_actors_test_set_errors_count = float(bad_actors_test_set_errors_count) / self._k_for_fold
            self._one_class_dict['#Errors_Bad_Actors_Test_Set'].append(bad_actors_test_set_errors_count)

            test_set_errors_stdev = self._calculate_stdev(bad_actors_test_set_errors)
            self._one_class_dict['STDEV_Bad_Actors_Errors_Test_Set'].append(test_set_errors_stdev)

            bad_actors_test_set_corrects_count = float(bad_actors_test_set_corrects_count) / self._k_for_fold
            self._one_class_dict['#Corrected_Bad_Actors_Test_Set'].append(bad_actors_test_set_corrects_count)

            test_set_corrects_stdev = self._calculate_stdev(bad_actors_test_set_corrects)
            self._one_class_dict['STDEV_Corrected_Bad_Actors_Test_Set'].append(test_set_corrects_stdev)

            good_actors_test_set_errors_count = float(good_actors_test_set_errors_count) / self._k_for_fold
            self._one_class_dict['#Errors_Good_Actors_Test_Set'].append(good_actors_test_set_errors_count)

            good_actors_test_set_errors_stdev = self._calculate_stdev(good_actors_test_set_errors)
            self._one_class_dict['STDEV_Good_Actors_Errors_Test_Set'].append(good_actors_test_set_errors_stdev)

            good_actors_test_set_corrects_count = float(good_actors_test_set_corrects_count) / self._k_for_fold
            self._one_class_dict['#Corrects_Good_Actors_Test_Set'].append(good_actors_test_set_corrects_count)

            good_actors_test_set_corrects_stdev = self._calculate_stdev(good_actors_test_set_corrects)
            self._one_class_dict['STDEV_Good_Actors_Corrects_Test_Set'].append(good_actors_test_set_corrects_stdev)

            test_set_errors_count = float(total_test_set_errors_count) / self._k_for_fold
            self._one_class_dict['#Errors_Test_Set'].append(test_set_errors_count)

            total_test_set_errors_stdev = self._calculate_stdev(total_test_set_errors)
            self._one_class_dict['STDEV_Errors_Test_Set'].append(total_test_set_errors_stdev)

            test_set_corrects_count = float(total_test_set_corrects_count) / self._k_for_fold
            self._one_class_dict['#Corrects_Test_Set'].append(test_set_corrects_count)

            total_test_set_corrects_stdev = self._calculate_stdev(total_test_set_corrects)
            self._one_class_dict['STDEV_Corrects_Test_Set'].append(total_test_set_corrects_stdev)

            self._one_class_dict['#Total_Test_Set'].append(
                isis_bad_actors_test_set_size_count + good_actors_test_set_size_count)
            # self._one_class_dict['#Total_Test_Set'].append(good_actors_test_set_size_count)

        one_class_result_dataframe = pd.DataFrame(self._one_class_dict, columns=self._one_class_column_names)

        full_path = self._path + "one_class_results.csv"
        # results_dataframe.to_csv(full_path)
        one_class_result_dataframe.to_csv(full_path, index=False)

        best_combination_elements = self._find_best_combination(one_class_result_dataframe)

        labeled_features_dataframe, unlabeled_features_dataframe = self._create_labeled_and_unlabeled_based_on_combination(
            best_combination_elements, feature_names,
            labeled_features_dataframe, unlabeled_features_dataframe)

        one_class_classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

        labeled_features_dataframe = labeled_features_dataframe.fillna(0)

        one_class_classifier.fit(labeled_features_dataframe)
        unlabeled_predictions = one_class_classifier.predict(unlabeled_features_dataframe)
        distances = one_class_classifier.decision_function(unlabeled_features_dataframe)

        self._write_predictions_into_file("one_class", str(len(best_combination_elements)),
                                          unlabeled_index_field_series,
                                          unlabeled_predictions, distances)

    def transfer_learning(self):
        for source_domain in self._source_domains:
            for target_domain in self._target_domains:
                for iteration in range(0, self._num_iterations):
                    print(iteration)
                    number_of_transfered_instances = 0
                    if self._source_input_type == 'csv':  # load source data from csv
                        source_df = pd.read_csv(self._source_input_path + '/' + source_domain + '.csv')
                    elif self._source_input_type == 'table':  # load from authors_features table
                        source_df = self._get_author_features_dataframe()

                    if self._target_input_type == 'csv':  # load target data from csv
                        target_df = pd.read_csv(self._target_input_path + '/' + target_domain + '.csv')
                    elif self._target_input_type == 'table':
                        target_df = self._get_author_features_dataframe()

                    # feature pre-processing
                    source_df = self._preprocess_dataframe(source_df)
                    target_df = self._preprocess_dataframe(target_df)

                    # find features in common
                    common_features = list(
                        set(source_df.columns) & set(target_df.columns))  # train and test must have the same features
                    source_df = source_df[common_features]
                    target_df = target_df[common_features]

                    # If we do not use instance based transfer learning we need to iterate only once through the num_neighbors loop
                    if not self._transfer_instances:
                        self._num_neighbors = [
                            -1]  # If we are not transferring instances iterate only once through the for loop

                    for k in self._num_neighbors:
                        if source_domain == target_domain:  # split 'target' dataset into train and test, ignore 'source'
                            msk = np.random.rand(len(target_df)) < (1 - self._target_train_test_split)
                            train_df = target_df[~msk]
                            test_df = target_df[msk]
                        elif source_domain != target_domain and not self._transfer_instances:  # train on dataset source and test on dataset target
                            train_df = source_df
                            test_df = target_df
                        elif source_domain != target_domain and self._transfer_instances:  # transfer knowledge from source to target

                            msk = np.random.rand(len(target_df)) < (1 - self._target_train_test_split)
                            train_df = target_df[~msk]
                            test_df = target_df[msk]
                            if self._target_train_percent_limit > 0:
                                train_df = train_df.sample(frac=self._target_train_percent_limit)
                            if self._target_test_percent_limit > 0:
                                test_df = test_df.sample(frac=self._target_test_percent_limit)

                            train_size_before_transfer = len(train_df)

                            if self._transfer_algo == 'BURAK':
                                '''
                                    B. Turhan, T. Menzies, A. B. Bener, and J. DiStefano.
                                    On the relative value of cross-company and within-company data for defect prediction.

                                    Burak Algorithm for instance-based transfer learning:
                                    The dataset we wish to improve or add additional data is called 'Target Dataset'.
                                    This algorithm first splits the target dataset into train and test sets.
                                    Then, for every object in the test set, it selects the k nearest neighbors wihtin any external 'Source dataset'
                                    and transfer these neighbors from the 'Source dataset' to the train set of the target dataset.
                                '''

                                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(source_df)
                                for index, row in test_df.iterrows():
                                    nbr_idx = nbrs.kneighbors(row, return_distance=False)
                                    for idx in nbr_idx[0]:
                                        train_df = train_df.append(source_df.iloc[idx])
                                train_df.drop_duplicates(inplace=True)
                                number_of_transfered_instances = len(train_df) - train_size_before_transfer
                            elif self._transfer_algo == 'GRAVITY_WEIGHTING':
                                '''
                                Ying Ma, Guangchun Luo, Xue Zeng, Aiguo Chen
                                Transfer learning for cross-company software defect prediction

                                Gravity Weighting: training instances are weighted inversely
                                proportional to their distance from the test instances, based
                                on measure of similarity defined in the paper
                                '''
                                train_df['weight'] = 1
                                min_values = test_df.min(axis=0)
                                max_values = test_df.max(axis=0)
                                for idx, row in source_df.iterrows():
                                    si = 0
                                    for col in target_df.columns:
                                        min_j = min_values[col]
                                        max_j = max_values[col]
                                        if min_j <= row[col] <= max_j:
                                            si += 1
                                    w = si / (len(source_df.columns) - si + 1)
                                    row['weight'] = w
                                    train_df = train_df.append(row)
                                X_train_weights = train_df.pop('weight').as_matrix()
                                train_df.drop_duplicates(inplace=True)
                                number_of_transfered_instances = len(train_df) - train_size_before_transfer

                        else:
                            raise Exception("Transfer learning module not configured properly")

                        X_train = train_df[source_df.columns.drop(self._targeted_class_name)]
                        y_train = train_df[self._targeted_class_name]

                        X_test = test_df[target_df.columns.drop(self._targeted_class_name)]
                        y_test = test_df[self._targeted_class_name]

                        # feature scaling
                        for scaling in self._feature_scaling:

                            if scaling == 'StandardScaler':
                                scaler = StandardScaler()
                            elif scaling == 'RobustScaler':
                                scaler = RobustScaler()
                            elif scaling == 'MinMaxScaler':
                                scaler = MinMaxScaler()

                            if scaling != 'None':
                                cols = list(X_train.columns)
                                X_train[cols] = scaler.fit_transform(X_train[cols].as_matrix())
                                X_test[cols] = scaler.fit_transform(X_test[cols].as_matrix())

                            # feature selection
                            for num_features in self._num_of_features_to_train:
                                for selection_method in self._feature_selection:
                                    if num_features == 'all':
                                        num_features = len(X_train.columns)

                                    selector = SelectKBest(score_func=globals()[selection_method],
                                                           k=int(num_features), )
                                    selector.fit_transform(X_train, y_train)
                                    scores = {X_train.columns[i]: selector.scores_[i] for i in
                                              range(len(X_train.columns))}
                                    filename = 'selected_features_source_' + source_domain + '_target_' + target_domain + '.csv'
                                    with open(filename, 'ab') as csv_file:
                                        writer = csv.writer(csv_file)
                                        writer.writerow(
                                            ['Feature Scaling Method', 'Num features', 'Feature Selection Method',
                                             'Feature', 'Value'])
                                        for key, value in list(scores.items()):
                                            writer.writerow([scaling, num_features, selection_method, key, value])

                                    sorted_features = sorted(scores, key=scores.get, reverse=True)[:int(num_features)]
                                    X_best_features_train = X_train[sorted_features]
                                    X_best_features_test = X_test[sorted_features]

                                    # model training
                                    trained_models = []
                                    for classifier_type_name in self._classifier_type_names:
                                        classifier = self._select_classifier_by_type(
                                            classifier_type_name=classifier_type_name)
                                        if source_domain != target_domain and self._transfer_instances and \
                                                (
                                                        self._transfer_algo == 'GRAVITY_WEIGHTING' or self._transfer_algo == 'MODIFIED_GRAVITY_WEIGHTING'):
                                            trained_model = classifier.fit(X=X_best_features_train, y=y_train,
                                                                           sample_weight=X_train_weights)
                                        else:
                                            trained_model = classifier.fit(X=X_best_features_train, y=y_train)

                                        trained_models.append(trained_model)

                                    one_time_flag = True
                                    for model in trained_models:
                                        model_name = model.__class__.__name__
                                        predictions_confidence = model.predict_proba(X_best_features_test)[:, 1]
                                        auc = roc_auc_score(y_test.values, predictions_confidence)
                                        predictions = model.predict(X_best_features_test)
                                        # conf = str(confusion_matrix(y_test.values, predictions))
                                        clasif_rep = precision_recall_fscore_support(y_test.values, predictions,
                                                                                     labels=[0, 1], pos_label=1)

                                        out_dict = {}

                                        if self._transfer_instances:
                                            out_dict['Transfer Learning'] = 'Transfer Learning'
                                            out_dict['Size of samples transferred'] = k
                                            out_dict['Algorithm'] = self._transfer_algo
                                        else:
                                            out_dict['Transfer Learning'] = 'No Transfer Learning'
                                            out_dict['Size of samples transferred'] = 0
                                            out_dict['Algorithm'] = 'No'

                                        out_dict['Source domain'] = source_domain
                                        out_dict['Target domain'] = target_domain
                                        out_dict['Number of Features'] = num_features
                                        out_dict['Feature Scaling Method'] = scaling
                                        out_dict['Feature Selection Method'] = selection_method
                                        if self._num_of_features == 'all':
                                            out_dict['Selected Features'] = 'all'
                                        else:
                                            out_dict['Selected Features'] = ', '.join(list(X_train.columns))

                                        out_dict['Number of Transfered Instances'] = number_of_transfered_instances
                                        if 'author_type' in train_df:
                                            if 0 in train_df['author_type'].value_counts():
                                                out_dict['Train Observations Good'] = int(
                                                    train_df['author_type'].value_counts()[0])
                                            else:
                                                out_dict['Train Observations Good'] = 0

                                            if 1 in train_df['author_type'].value_counts():
                                                out_dict['Train Observations Bad'] = int(
                                                    train_df['author_type'].value_counts()[1])
                                            else:
                                                out_dict['Train Observations Bad'] = 0
                                        else:
                                            out_dict['Train Observations Good'] = 0
                                            out_dict['Train Observations Bad'] = 0

                                        if 'author_type' in test_df:
                                            out_dict['Test Observations Good'] = int(
                                                test_df['author_type'].value_counts()[0])
                                            out_dict['Test Observations Bad'] = int(
                                                test_df['author_type'].value_counts()[1])
                                        else:
                                            out_dict['Test Observations Good'] = 0
                                            out_dict['Test Observations Bad'] = 0

                                        out_dict['Precision Good'] = clasif_rep[0][0].round(2)
                                        out_dict['Precision Bad'] = clasif_rep[0][1].round(2)
                                        out_dict['Recall Good'] = clasif_rep[1][0].round(2)
                                        out_dict['Recall Bad'] = clasif_rep[1][1].round(2)
                                        out_dict['F1-score Good'] = clasif_rep[2][0].round(2)
                                        out_dict['F1-score Bad'] = clasif_rep[2][1].round(2)

                                        tn, fp, fn, tp = confusion_matrix(y_test.values, predictions).ravel()

                                        out_dict['TP'] = tp
                                        out_dict['TN'] = tn
                                        out_dict['FP'] = fp
                                        out_dict['FN'] = fn

                                        out_dict['AUC'] = auc
                                        out_df = pd.DataFrame(out_dict, index=[model_name],
                                                              columns=['Source domain', 'Target domain',
                                                                       'Transfer Learning',
                                                                       'Size of samples transferred', 'Algorithm',
                                                                       'Feature Scaling Method', 'Number of Features',
                                                                       'Feature Selection Method', 'Selected Features',
                                                                       'Number of Transfered Instances',
                                                                       'Train Observations Good',
                                                                       'Train Observations Bad',
                                                                       'Test Observations Good',
                                                                       'Test Observations Bad', 'Precision Good',
                                                                       'Precision Bad', 'Recall Good',
                                                                       'Recall Bad', 'F1-score Good', 'F1-score Bad',
                                                                       'TP', 'TN', 'FP', 'FN', 'AUC'])

                                        tl = 'transfer_learning_' + out_dict['Transfer Learning']
                                        filename = 'results_' + tl + '__source_' + source_domain + '_target_' + target_domain + '.csv'
                                        if one_time_flag:
                                            out_df.to_csv(filename, mode='a', header=True)
                                            one_time_flag = False
                                        else:
                                            out_df.to_csv(filename, mode='a', header=False)
                                        print(model_name)
                                        print('AUC: ' + str(auc))
                                        print('Confusion: \n' + str(
                                            confusion_matrix(y_test.values, predictions, labels=[1, 0])))

                                        print('##################################################')

    def _preprocess_dataframe(self, df):
        df = self._remove_features(self._removed_features, df)  # remove author_sub_type, user screen_name, etc.
        df = replace_nominal_class_to_numeric(df, self._optional_classes)  # replace 'bad_actor' to 1
        df.replace('?', np.NaN, inplace=True)  # replace ? for nan
        df.dropna(axis=0, how='any', subset=[self._targeted_class_name],
                  inplace=True)  # drop row if target class is NaN
        df.dropna(axis=1, how='all', inplace=True)  # drop column if all values  are nan
        df = df.apply(pd.to_numeric)  # convert all values to numeric
        df = df.drop(df.std()[df.std() < self._stdev_threshold].index.values, axis=1)

        if self._replace_missing_values == 'mean':  # replace missing values with column mean or 0
            df.fillna(df.mean(), inplace=True)
        else:
            df.fillna(0, inplace=True)
        return df

    def _create_target_class_classifier_dictionary(self):
        # Dictionary = Dict["author_type"]["RandomForest"][5]["AUC"] = 0.99
        start_time = time.time()
        print(
            "_create_target_class_classifier_dictionary started for " + self.__class__.__name__ + " started at " + str(
                start_time))
        performance_measure_dictionary = {}
        for performance_measure in self._performance_measures:
            performance_measure_dictionary[performance_measure] = 0

        num_of_features_performance_measure_dictionary = {}
        for num_of_features in self._num_of_features_to_train:
            deep_copy_results_dictionary = copy.deepcopy(performance_measure_dictionary)
            num_of_features_performance_measure_dictionary[num_of_features] = deep_copy_results_dictionary

        classifier_performance_measure_dictionary = {}
        for classifier_type_name in self._classifier_type_names:
            deep_copy_results_dictionary = copy.deepcopy(num_of_features_performance_measure_dictionary)
            classifier_performance_measure_dictionary[classifier_type_name] = deep_copy_results_dictionary

        target_class_classifier_dictionary = {}

        deep_copy_classifier_performance_measure_dictionary = copy.deepcopy(classifier_performance_measure_dictionary)
        target_class_classifier_dictionary[
            self._targeted_class_name] = deep_copy_classifier_performance_measure_dictionary

        end_time = time.time()
        print("_create_target_class_classifier_dictionary started for " + self.__class__.__name__ + " ended at " + str(
            end_time))
        return target_class_classifier_dictionary

    def _get_author_features_dataframe(self):
        start_time = time.time()
        print(
            "_get_author_features_dataframe started for " + self.__class__.__name__ + " started at " + str(start_time))
        data_frame_creator = DataFrameCreator(self._db)
        data_frame_creator.create_author_features_data_frame()
        author_features_dataframe = data_frame_creator.get_author_features_data_frame()

        end_time = time.time()
        print("_get_author_features_dataframe ended for " + self.__class__.__name__ + " ended at " + str(end_time))
        return author_features_dataframe

    def _write_results_into_file(self):
        start_time = time.time()
        print("_write_results_into_file started for " + self.__class__.__name__ + " started at " + str(start_time))

        full_path_file_name = self._path + self._results_file_name
        if not os.path.exists(full_path_file_name):
            open(full_path_file_name, 'w').close()

        with open(full_path_file_name, "w") as text_file:
            text_file.write("Supervised learning results:" + "\n")
            text_file.write("----------------------------------------------------------" + "\n")
            text_file.write("Target class name: {0}".format(self._targeted_class_name) + "\n")
            for classifier_type_name in self._classifier_type_names:
                text_file.write("Selected classifier: {0}".format(classifier_type_name) + "\n")
                for num_of_features in self._num_of_features_to_train:
                    text_file.write("Num of features: {0}".format(num_of_features) + "\n")
                    self._print_correctly_and_not_correctly_instances(classifier_type_name, num_of_features, text_file)

                    for performance_measure in self._performance_measures:
                        performance_measure_result = \
                            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                                num_of_features][performance_measure]
                        if performance_measure == PerformanceMeasures.CONFUSION_MATRIX:
                            text_file.write(performance_measure + ":" + "\n")
                            text_file.write("{0}".format(performance_measure_result) + "\n")

                        elif performance_measure == PerformanceMeasures.SELECTED_FEATURES:
                            text_file.write(performance_measure + ":" + "\n")
                            text_file.write(', '.join(performance_measure_result) + "\n")

                        else:
                            text_file.write(performance_measure + ": {0}".format(performance_measure_result) + "\n")
                    text_file.write("----------------------------------------------------------" + "\n")

            text_file.write("End of supervised learning results:" + "\n")
            text_file.write("----------------------------------------------------------" + "\n")

        end_time = time.time()
        print("_write_results_into_file ended for " + self.__class__.__name__ + " started at " + str(end_time))

    def _prepare_dataframe_for_learning(self, dataframe):
        start_time = time.time()
        print(
            "_prepare_dataframe_for_learning started for " + self.__class__.__name__ + " started at " + str(start_time))

        # dataframe.reset_index(drop=True, inplace=True)

        # Replace None in 0 for later calculation
        if self._replace_missing_values == 'zero':
            dataframe = dataframe.fillna(0)
        elif self._replace_missing_values == 'mean':
            dataframe.fillna(dataframe.mean(), inplace=True)

        # # Replace nominal class in numeric classes
        # num_of_class = len(self._optional_classes)
        # for i in range(num_of_class):
        #     class_name = self._optional_classes[i]
        #     dataframe = dataframe.replace(to_replace=class_name, value=i)

        dataframe = replace_nominal_class_to_numeric(dataframe, self._optional_classes)

        # dataframe = dataframe.replace(to_replace=Author_Type.GOOD_ACTOR, value=0)
        # dataframe = dataframe.replace(to_replace=Author_Type.BAD_ACTOR, value=1)
        #
        # dataframe = dataframe.replace(to_replace=Author_Subtype.PRIVATE, value=0)
        # dataframe = dataframe.replace(to_replace=Author_Subtype.COMPANY, value=1)
        # dataframe = dataframe.replace(to_replace=Author_Subtype.NEWS_FEED, value=2)
        # dataframe = dataframe.replace(to_replace=Author_Subtype.SPAMMER, value=3)
        # dataframe = dataframe.replace(to_replace=Author_Subtype.BOT, value=4)
        # dataframe = dataframe.replace(to_replace=Author_Subtype.CROWDTURFER, value=5)
        # dataframe = dataframe.replace(to_replace=Author_Subtype.ACQUIRED, value=6)

        index_field_series = dataframe.pop(self._index_field)
        targeted_class_series = dataframe.pop(self._targeted_class_name)

        # Remove unnecessary features
        dataframe = self._remove_features(self._removed_features, dataframe)

        num_of_features = len(self._selected_features)
        if num_of_features > 0:
            selected_feature = self._selected_features[0]
            selected_feature_series = dataframe[selected_feature]
            targeted_dataframe = pd.DataFrame(selected_feature_series, columns=[selected_feature])

            for i in range(1, num_of_features):
                selected_feature = self._selected_features[i]
                targeted_dataframe[selected_feature] = dataframe[selected_feature]

        else:
            targeted_dataframe = dataframe

        return targeted_dataframe, targeted_class_series, index_field_series

    def _update_performance_measures(self, classifier_type_name, num_of_features, predictions, predictions_proba,
                                     performance_measure, test_class, classes):

        num_of_unique_classes = len(test_class.unique())

        if performance_measure is PerformanceMeasures.AUC:
            if num_of_unique_classes == 2:
                auc_score = roc_auc_score(test_class, predictions_proba[:, 1])
                print("roc_auc_score is: " + str(auc_score))
                self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                    num_of_features][performance_measure] \
                    += roc_auc_score(test_class, predictions_proba[:, 1])


            elif num_of_unique_classes > 2:
                real_array = self._enumerate_label(classes, test_class)
                auc_score = self._calculate_weighted_auc(performance_measure, real_array, predictions_proba)
                self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                    num_of_features][performance_measure] += \
                    auc_score



        elif performance_measure is PerformanceMeasures.PRECISION:
            if len(classes) > 2:
                precision = precision_score(test_class, predictions, average='micro')
            else:
                precision = precision_score(test_class, predictions)
            print("precision_score is: " + str(precision))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += precision


        elif performance_measure is PerformanceMeasures.RECALL:
            if len(classes) > 2:
                recall = recall_score(test_class, predictions, average='micro')
            else:
                recall = recall_score(test_class, predictions)
            print("accuracy_score is: " + str(recall))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += recall


        elif performance_measure is PerformanceMeasures.ACCURACY:
            accuracy = accuracy_score(test_class, predictions)
            print("accuracy_score is: " + str(accuracy))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += accuracy


        elif performance_measure is PerformanceMeasures.CONFUSION_MATRIX:
            score = confusion_matrix(test_class, predictions)
            print("confusion_matrix is: " + str(score))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += confusion_matrix(test_class, predictions)

        elif performance_measure is PerformanceMeasures.LRAP:
            test_class_new = []
            for prediction in test_class:
                multi_class_prediction = [0] * len(classes)
                multi_class_prediction[prediction] = 1
                test_class_new.append(multi_class_prediction)
            score = label_ranking_average_precision_score(test_class_new, predictions_proba)
            print("LRAP is: " + str(score))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += score

        elif performance_measure is PerformanceMeasures.LRL:
            test_class_new = []
            for prediction in test_class:
                multi_class_prediction = [0] * len(classes)
                multi_class_prediction[prediction] = 1
                test_class_new.append(multi_class_prediction)
            score = label_ranking_loss(test_class_new, predictions_proba)
            print("LRL is: " + str(score))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += score

        elif performance_measure is PerformanceMeasures.spearman_correlation:
            score = stats.spearmanr(test_class, predictions).correlation
            print("spearman_correlation is: " + str(score))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += score

        elif performance_measure is PerformanceMeasures.spearman_coefficient_p_value:
            score = stats.spearmanr(test_class, predictions).pvalue
            print("spearman_coefficient_p_value is: " + str(score))
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                performance_measure] \
                += score

    def _calculate_average_for_performance_measures(self, classifier_type_name, valid_k, num_of_features):
        for performance_measure in self._performance_measures:
            # if not performance_measure == PerformanceMeasures.SELECTED_FEATURES and \
            #         not(target_class_name == self._author_sub_type_class_name
            #             and (performance_measure == PerformanceMeasures.AUC)):
            if not (performance_measure == PerformanceMeasures.SELECTED_FEATURES or
                    performance_measure == PerformanceMeasures.CONFUSION_MATRIX or
                    performance_measure == PerformanceMeasures.CORRECTLY_CLASSIFIED or
                    performance_measure == PerformanceMeasures.INCORRECTLY_CLASSIFIED):
                self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                    num_of_features][performance_measure] /= valid_k

    def _enumerate_label(self, classes, test_real):
        classesDict = {}
        for i, c in enumerate(classes):
            classesDict[c] = i
        realArray = [classesDict[x] for x in test_real]
        return realArray

    def _calculate_weighted_auc(self, performance_measure_name, real_array, predictions_proba):
        num_of_classes = len(set(real_array))
        total = len(predictions_proba)
        weighted_auc_score = 0.0
        performance_measure_score = 0
        for i in range(num_of_classes):
            curr_real = []
            curr_preds = []
            pos_count = 0.0
            for j, value in enumerate(real_array):
                if value == i:
                    curr_real.append(1)
                    pos_count += 1
                else:
                    curr_real.append(0)
                curr_preds.append(predictions_proba[j][i])
            weight = pos_count / total
            if performance_measure_name == PerformanceMeasures.AUC:
                performance_measure_score = roc_auc_score(curr_real, curr_preds, average="weighted")
            weighted_auc_score += performance_measure_score * weight
        return weighted_auc_score

    def _reduce_dimensions_by_num_of_features(self, labeled_author_features_dataframe, targeted_class_series,
                                              num_of_features):
        print("Create dataframe with the {0} best features".format(num_of_features))

        return self._find_k_best_and_reduce_dimensions(num_of_features, labeled_author_features_dataframe,
                                                       targeted_class_series)

    def _get_k_best_feature_names(self, k_best_classifier, original_dataframe):
        mask = k_best_classifier.get_support()
        best_feature_names = []
        column_names = list(original_dataframe.columns.values)
        for boolean_value, feature_name in zip(mask, column_names):
            if boolean_value == True:
                best_feature_names.append(feature_name)
        return best_feature_names

    def _select_classifier_by_type(self, classifier_type_name):
        selected_classifier = None

        if classifier_type_name == Classifiers.RandomForest:
            selected_classifier = RandomForestClassifier(n_estimators=100)

        elif classifier_type_name == Classifiers.DecisionTree:
            selected_classifier = tree.DecisionTreeClassifier()

        elif classifier_type_name == Classifiers.AdaBoost:
            selected_classifier = AdaBoostClassifier(n_estimators=30)

        elif classifier_type_name == Classifiers.XGBoost:
            selected_classifier = xgb.XGBClassifier()

        return selected_classifier

    def _save_trained_model(self, selected_classifier, classifier_type_name, num_of_features,
                            reduced_dataframe_column_names):
        if not os.path.exists(self._full_path_model_directory):
            os.makedirs(self._full_path_model_directory)

        # save model
        full_model_file_path = self._full_path_model_directory + "trained_classifier_" + self._targeted_class_name + "_" + \
                               classifier_type_name + "_" + str(
            num_of_features) + "_features.pkl"
        joblib.dump(selected_classifier, full_model_file_path)

        # save features
        model_features_file_path = self._full_path_model_directory + "trained_classifier_" + self._targeted_class_name + "_" + classifier_type_name + "_" + str(
            num_of_features) + "_selected_features.pkl"
        joblib.dump(reduced_dataframe_column_names, model_features_file_path)

    def _find_k_best_and_reduce_dimensions(self, num_of_features, labeled_author_features_dataframe,
                                           targeted_class_series):
        k_best_classifier = SelectKBest(score_func=f_classif, k=num_of_features)

        k_best_classifier = k_best_classifier.fit(labeled_author_features_dataframe, targeted_class_series)
        k_best_features = k_best_classifier.fit_transform(labeled_author_features_dataframe,
                                                          targeted_class_series)

        reduced_dataframe_column_names = self._get_k_best_feature_names(k_best_classifier,
                                                                        labeled_author_features_dataframe)

        print("Best features found are: ")
        print(', '.join(reduced_dataframe_column_names))

        reduced_dataframe = pd.DataFrame(k_best_features, columns=reduced_dataframe_column_names)

        return reduced_dataframe, reduced_dataframe_column_names

    def _print_correctly_and_not_correctly_instances(self, classifier_type_name, num_of_features, text_file):
        num_of_correct_instances, num_of_incorrect_instances = self._calculate_correctly_and_not_correctly_instances(
            classifier_type_name, num_of_features)

        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
            PerformanceMeasures.CORRECTLY_CLASSIFIED] = num_of_correct_instances
        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
            PerformanceMeasures.INCORRECTLY_CLASSIFIED] = num_of_incorrect_instances

        text_file.write("Correctly classified instances: {0}".format(num_of_correct_instances) + "\n")
        text_file.write("Incorrectly classified instances: {0}".format(num_of_incorrect_instances) + "\n")

        total_instances = num_of_correct_instances + num_of_incorrect_instances
        text_file.write("Total number of instances: {0}".format(total_instances) + "\n")

    def _calculate_correctly_and_not_correctly_instances(self, classifier_type_name, num_of_features):
        num_of_correct_instances = 0
        num_of_incorrect_instances = 0

        confusion_matrix_result = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.CONFUSION_MATRIX]
        dimension = confusion_matrix_result.shape[0]
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    num_of_correct_instances += confusion_matrix_result[i][j]
                else:
                    num_of_incorrect_instances += confusion_matrix_result[i][j]

        return num_of_correct_instances, num_of_incorrect_instances

    def _set_selected_features_into_dictionary(self, classifier_type_name, num_of_features,
                                               dataframe_column_names):

        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
            PerformanceMeasures.SELECTED_FEATURES] = dataframe_column_names

    def _select_valid_k(self, targeted_class_series):
        valid_k = retreive_valid_k(self._k_for_fold, targeted_class_series)
        k_folds = StratifiedKFold(targeted_class_series, valid_k)
        return k_folds, valid_k

    def _create_train_and_test_dataframes_and_classes(self, targeted_dataframe, train_indexes,
                                                      test_indexes, targeted_class_series):
        train_set_dataframe = targeted_dataframe.loc[train_indexes.tolist()]
        test_set_dataframe = targeted_dataframe.loc[test_indexes.tolist()]
        train_class = targeted_class_series[train_indexes]
        test_class = targeted_class_series[test_indexes]
        return train_set_dataframe, test_set_dataframe, train_class, test_class

    def _retreive_unlabeled_authors_dataframe(self, dataframe):
        # unlabeled_data_frame = dataframe[dataframe.author_type.isnull()]
        unlabeled_data_frame = dataframe.loc[dataframe[self._targeted_class_name].isnull()]
        return unlabeled_data_frame

    def _reduce_dataframe_dimenstions_by_reduced_column_names(self, unlabeled_dataframe, reduced_column_names):
        column_names = unlabeled_dataframe.columns.values
        column_names_dictionary = dict.fromkeys(reduced_column_names)
        for column_name in column_names:
            if column_name not in column_names_dictionary:
                unlabeled_dataframe.pop(column_name)
        return unlabeled_dataframe

    def _replace_predictions_class_from_int_to_string(self, predictions_series):
        predictions_series = self._replace_numeric_class_to_nominal(predictions_series)
        return predictions_series

    def _write_results_as_table(self):
        rows = []
        for classifier_type_name in self._classifier_type_names:
            for num_of_features in self._num_of_features_to_train:
                row = self._create_row_result(classifier_type_name, num_of_features)
                rows.append(row)

        dataframe = pd.DataFrame(rows, columns=self._column_names_for_results_table)

        path = self._path + self._results_table_file_name
        dataframe.to_csv(path, index=False)

    def _set_dataframe_columns_types(self, df):
        column_names = df.columns.values
        for column_name in column_names:
            print("feature_name: " + column_name)
            feature_series = df[column_name]
            feature_series = feature_series.astype(np.float64)
            # feature_series = feature_series.astype(np.int64)
            df[column_name] = feature_series
        return df

    def _create_row_result(self, classifier_type_name, num_of_features):
        row = []
        num_of_correctly_instances = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.CORRECTLY_CLASSIFIED]
        num_of_incorrectly_instances = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.INCORRECTLY_CLASSIFIED]
        total_instances = num_of_correctly_instances + num_of_incorrectly_instances

        auc = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.AUC]
        accuracy = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.ACCURACY]
        precision = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.PRECISION]
        recall = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.RECALL]
        lrap = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.LRAP]
        lrl = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.LRL]
        spearman_correlation = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.spearman_correlation]
        spearman_coefficient_p_value = \
            self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][
                PerformanceMeasures.spearman_coefficient_p_value]

        row.append(self._targeted_class_name)
        row.append(classifier_type_name)
        row.append(num_of_features)
        row.append(num_of_correctly_instances)
        row.append(num_of_incorrectly_instances)
        row.append(total_instances)
        row.append(auc)
        row.append(accuracy)
        row.append(precision)
        row.append(recall)
        row.append(lrap)
        row.append(lrl)
        row.append(spearman_correlation)
        row.append(spearman_coefficient_p_value)

        return row

    def _init_max_classifier_dict(self):
        dictionary = {}
        dictionary["Num_of_features"] = -1
        dictionary["Classifier_type_name"] = ""
        dictionary[PerformanceMeasures.AUC] = -1
        dictionary[PerformanceMeasures.ACCURACY] = -1
        dictionary[PerformanceMeasures.RECALL] = -1
        dictionary[PerformanceMeasures.PRECISION] = -1

        dictionary["Best_feature_names"] = ""

        max_classifier_dict = {}

        deep_copy_dictionary = copy.deepcopy(dictionary)
        max_classifier_dict[self._targeted_class_name] = deep_copy_dictionary

        return max_classifier_dict

    def _set_max_classifier(self, classifier_type_name, num_of_features, dataframe_column_names,
                            current_auc, current_accuracy, current_precision, current_recall):
        self._max_classifier_dictionary[self._targeted_class_name]["Num_of_features"] = num_of_features
        self._max_classifier_dictionary[self._targeted_class_name]["Classifier_type_name"] = classifier_type_name
        self._max_classifier_dictionary[self._targeted_class_name]["Best_feature_names"] = dataframe_column_names
        self._max_classifier_dictionary[self._targeted_class_name][PerformanceMeasures.AUC] = current_auc
        self._max_classifier_dictionary[self._targeted_class_name][PerformanceMeasures.ACCURACY] = current_accuracy
        self._max_classifier_dictionary[self._targeted_class_name][PerformanceMeasures.PRECISION] = current_precision
        self._max_classifier_dictionary[self._targeted_class_name][PerformanceMeasures.RECALL] = current_recall

    def _create_best_classifier_train_save_and_predict(self, labeled_author_features_dataframe,
                                                       targeted_class_series, unlabeled_author_dataframe,
                                                       unlabeled_index_field_series, unlabeled_targeted_class_series):

        self._find_max_average_auc_classifier()

        selected_classifier = self._create_best_classifier_and_train(labeled_author_features_dataframe,
                                                                     targeted_class_series)

        num_of_features = self._max_classifier_dictionary[self._targeted_class_name]["Num_of_features"]
        classifier_type_name = self._max_classifier_dictionary[self._targeted_class_name]["Classifier_type_name"]
        best_feature_names = self._max_classifier_dictionary[self._targeted_class_name]["Best_feature_names"]

        self._save_trained_model(selected_classifier, classifier_type_name, num_of_features, best_feature_names)

        unlabeled_author_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
            unlabeled_author_dataframe, unlabeled_targeted_class_series, num_of_features)

        predictions_series, predictions_proba_series = self._predict_classifier(selected_classifier,
                                                                                unlabeled_author_dataframe)

        self._write_predictions_into_file(classifier_type_name, num_of_features,
                                          unlabeled_index_field_series, predictions_series, predictions_proba_series)

    def _create_best_classifier_and_train(self, labeled_author_features_dataframe, targeted_class_series):
        num_of_features = self._max_classifier_dictionary[self._targeted_class_name]["Num_of_features"]

        labeled_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
            labeled_author_features_dataframe, targeted_class_series, num_of_features)

        classifier_type_name = self._max_classifier_dictionary[self._targeted_class_name]["Classifier_type_name"]

        selected_classifier = self._select_classifier_by_type(classifier_type_name)

        selected_classifier.fit(labeled_dataframe, targeted_class_series)

        return selected_classifier

    def _predict_classifier(self, selected_classifier, unlabeled_author_dataframe):
        predictions = selected_classifier.predict(unlabeled_author_dataframe)
        predictions_series = pd.Series(predictions)

        predictions_series = self._replace_predictions_class_from_int_to_string(predictions_series)

        predictions_proba = selected_classifier.predict_proba(unlabeled_author_dataframe)
        num_of_classes = len(self._optional_classes)
        if num_of_classes == 2:
            predictions_proba_series = pd.Series(predictions_proba[:, 1])
        elif num_of_classes > 2:
            predictions_proba_ndarray = np.array(predictions_proba)
            max_predictions_proba = predictions_proba_ndarray.max(axis=1)
            predictions_proba_series = pd.Series(max_predictions_proba)
        return predictions_series, predictions_proba_series

    def _write_predictions_into_file(self, classifier_type_name, num_of_features,
                                     unlabeled_index_field_series, predictions_series, predictions_proba_series):

        unlabeled_dataframe_with_prediction = pd.DataFrame(unlabeled_index_field_series,
                                                           columns=[self._index_field])

        unlabeled_dataframe_with_prediction.reset_index(drop=True, inplace=True)
        unlabeled_dataframe_with_prediction["predicted"] = predictions_series
        unlabeled_dataframe_with_prediction["prediction"] = predictions_proba_series

        full_path = self._path + "predictions_on_unlabeled_authors_" + self._targeted_class_name + "_" + \
                    classifier_type_name + "_" + str(num_of_features) + "_features.csv"
        # results_dataframe.to_csv(full_path)
        unlabeled_dataframe_with_prediction.to_csv(full_path, index=False)

        table_name = "unlabeled_predictions"
        self._db.drop_unlabeled_predictions(table_name)

        engine = self._db.engine
        unlabeled_dataframe_with_prediction.to_sql(name=table_name, con=engine)

    def _find_max_average_auc_classifier(self):
        max_auc = -1
        for classifier_type_name in self._classifier_type_names:
            for num_of_features in self._num_of_features_to_train:
                current_auc = \
                    self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                        num_of_features][
                        PerformanceMeasures.AUC]
                if max_auc < current_auc:
                    max_auc = current_auc
                    current_accuracy = \
                        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                            num_of_features][PerformanceMeasures.ACCURACY]
                    current_precision = \
                        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                            num_of_features][PerformanceMeasures.PRECISION]
                    current_recall = \
                        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                            num_of_features][PerformanceMeasures.RECALL]
                    selected_features = \
                        self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][
                            num_of_features][PerformanceMeasures.SELECTED_FEATURES]

                    self._set_max_classifier(classifier_type_name, num_of_features, selected_features,
                                             current_auc, current_accuracy, current_precision, current_recall)

    def _get_trained_classifier(self):
        full_path_selected_model = self._full_path_model_directory + self._prepared_classifier_name

        trained_classifier = joblib.load(full_path_selected_model)
        return trained_classifier

    def _get_selected_features(self):
        full_path_selected_features = self._full_path_model_directory + self._selected_features_to_load

        selected_features = joblib.load(full_path_selected_features)
        return selected_features

    def _reduce_unlabeled_dataframe_dimensions_and_predict(self):
        pass

    def _remove_features(self, features_to_remove, dataframe):
        '''
        This function is responsible to remove features.
        :param dataframe:
        :return:dataframe without removed columns
        '''
        # print("Start remove_unnecessary_features")
        # Remove unnecessary features
        dataframe_columns = dataframe.columns
        for unnecessary_feature in features_to_remove:
            if unnecessary_feature in dataframe_columns:
                dataframe.pop(unnecessary_feature)

        return dataframe

    # def _replace_nominal_class_to_numeric(self, dataframe):
    #     num_of_class = len(self._optional_classes)
    #     for i in range(num_of_class):
    #         class_name = self._optional_classes[i]
    #         dataframe = dataframe.replace(to_replace=class_name, value=i)
    #     return dataframe

    def _replace_numeric_class_to_nominal(self, dataframe):
        num_of_class = len(self._optional_classes)
        for i in range(num_of_class):
            class_name = self._optional_classes[i]
            dataframe = dataframe.replace(to_replace=i, value=class_name)
        return dataframe

    def _find_best_combination(self, one_class_result_dataframe):
        max_series = one_class_result_dataframe.loc[one_class_result_dataframe['#Corrects_Test_Set'].idxmax()]
        best_combination_name = max_series["Combination"]
        best_combination_elements = best_combination_name.split("+")
        return best_combination_elements

    def _create_one_class_dictionary(self):
        one_class_dict = {}
        for column_name in self._one_class_column_names:
            one_class_dict[column_name] = []
        return one_class_dict

    def _create_labeled_and_unlabeled_based_on_combination(self, best_combination_elements, feature_names,
                                                           labeled_features_dataframe, unlabeled_features_dataframe):
        best_combination_set = set(best_combination_elements)
        feature_names_set = set(feature_names)
        features_to_remove_set = feature_names_set - best_combination_set
        features_to_remove = list(features_to_remove_set)

        labeled_features_dataframe = self._remove_features(features_to_remove, labeled_features_dataframe)
        unlabeled_features_dataframe = self._remove_features(features_to_remove, unlabeled_features_dataframe)

        return labeled_features_dataframe, unlabeled_features_dataframe

    def _create_labeled_and_unlabeled_based_on_author_features(self):
        # author_features_dataframe = self._get_author_features_dataframe()
        #
        # unlabeled_features_dataframe = self._retreive_unlabeled_authors_dataframe(author_features_dataframe)
        # unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_index_field_series = \
        #     self._prepare_dataframe_for_learning(unlabeled_features_dataframe)
        #
        # labeled_features_dataframe = retreive_labeled_authors_dataframe(self._targeted_class_name,
        #                                                                 author_features_dataframe)
        labeled_features_dataframe, unlabeled_features_dataframe, unlabeled_targeted_class_series, \
        unlabeled_index_field_series = self._create_unlabeled_authors_dataframe_and_raw_labeled_authors_dataframe()
        labeled_features_dataframe, targeted_class_series, index_field_series = \
            self._prepare_dataframe_for_learning(labeled_features_dataframe)

        return labeled_features_dataframe, unlabeled_features_dataframe, targeted_class_series, unlabeled_targeted_class_series, unlabeled_index_field_series

    def _create_unlabeled_authors_dataframe_and_raw_labeled_authors_dataframe(self):
        author_features_dataframe = self._get_author_features_dataframe()

        unlabeled_features_dataframe = self._retreive_unlabeled_authors_dataframe(author_features_dataframe)
        unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_index_field_series = \
            self._prepare_dataframe_for_learning(unlabeled_features_dataframe)

        labeled_features_dataframe = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                        author_features_dataframe)
        return labeled_features_dataframe, unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_index_field_series

    def _calculate_stdev(self, records):
        records_numpy_array = numpy.array(records)
        stdev = numpy.std(records_numpy_array, axis=0)
        return stdev

    def _train_one_class_classifiers_for_each_combination(self, labeled_features_dataframe, targeted_class_series):
        feature_names = list(labeled_features_dataframe.columns.values)

        # create all combinations for finding the best classifier
        optional_combinations = sum([list(map(list, combinations(feature_names, i))) for i in range(len(feature_names) + 1)],
                                    [])
        i = 0
        for combination in optional_combinations:
            i += 1
            print("Combination: {0} {1}/{2}".format(combination, i, len(optional_combinations)))
            if len(combination) > 0:
                # self._calculate_combination(combination, feature_names, labeled_features_dataframe, targeted_class_series)
                training_set_size_count = 0
                test_set_size_count = 0

                test_set_errors_count = 0
                test_set_errors = []
                test_set_corrects_count = 0
                test_set_corrects = []
                test_set_total = 0

                combination_name = "+".join(combination)
                self._one_class_dict['Combination'].append(combination_name)

                features_to_remove = self._calculate_features_to_remove(combination, feature_names)

                k_folds, valid_k = self._select_valid_k(targeted_class_series)
                for train_indexes, test_indexes in k_folds:
                    train_set_dataframe, test_set_dataframe, train_class, test_class = self._create_train_and_test_dataframes_and_classes(
                        labeled_features_dataframe,
                        train_indexes, test_indexes,
                        targeted_class_series)

                    train_set_dataframe = self._remove_features(features_to_remove, train_set_dataframe)
                    test_set_dataframe = self._remove_features(features_to_remove, test_set_dataframe)

                    training_size = train_set_dataframe.shape[0]
                    training_set_size_count += training_size

                    test_set_size = test_set_dataframe.shape[0]
                    test_set_size_count += test_set_size

                    one_class_classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

                    one_class_classifier.fit(train_set_dataframe)

                    test_set_predictions = one_class_classifier.predict(test_set_dataframe)
                    # distances = one_class_classifier.decision_function(test_set_dataframe)
                    num_error_test = test_set_predictions[test_set_predictions == -1].size
                    test_set_errors.append(num_error_test)
                    test_set_errors_count += num_error_test
                    test_set_total += num_error_test

                    num_correct_test = test_set_predictions[test_set_predictions == 1].size
                    test_set_corrects.append(num_correct_test)
                    test_set_corrects_count += num_correct_test
                    test_set_total += num_correct_test

                training_set_size_count = float(training_set_size_count) / self._k_for_fold
                self._one_class_dict['#Bad_Actors_Training_Set'].append(training_set_size_count)

                test_set_errors_count = float(test_set_errors_count) / self._k_for_fold
                self._one_class_dict['#Errors_Test_Set'].append(test_set_errors_count)

                test_set_errors_stdev = self._calculate_stdev(test_set_errors)
                self._one_class_dict['STDEV_Errors_Test_Set'].append(test_set_errors_stdev)

                test_set_corrects_count = float(test_set_corrects_count) / self._k_for_fold
                self._one_class_dict['#Corrected_Test_Set'].append(test_set_corrects_count)

                test_set_corrects_stdev = self._calculate_stdev(test_set_corrects)
                self._one_class_dict['STDEV_Corrected_Test_Set'].append(test_set_corrects_stdev)

                self._one_class_dict['#Total_Test_Set'].append(test_set_size_count)

    def _calculate_features_to_remove(self, combination, feature_names):
        combination_set = set(combination)
        feature_names_set = set(feature_names)
        features_to_remove_set = feature_names_set - combination_set
        features_to_remove = list(features_to_remove_set)
        return features_to_remove

    def _create_labeled_dfs(self, labeled_authors_df):
        targeted_class_dfs = []
        for optional_class in self._optional_classes:
            target_class_labeled_authors_df = labeled_authors_df.loc[
                labeled_authors_df[self._targeted_class_name] == optional_class]
            targeted_class_dfs.append(target_class_labeled_authors_df)
        return targeted_class_dfs

    def _build_training_set(self, training_size_percent, targeted_class_dfs):
        sample_targeted_class_dfs = []
        for targeted_class_df in targeted_class_dfs:
            # Choosing randonly samples from each class
            sample_targeted_class_df = targeted_class_df.sample(frac=training_size_percent)
            sample_targeted_class_dfs.append(sample_targeted_class_df)
        training_set_df = pd.concat(sample_targeted_class_dfs)
        return training_set_df

    def _replicate_authors_by_num_of_labeled_claim_posts(self, training_df, author_guid_binary_verdict_results_df):
        replicated_training_df = pd.DataFrame()
        i = 0
        for author_guid, row in training_df.iterrows():
            i += 1
            msg = "\r Replicating authors {0}/{1}".format(i, len(training_df.index))
            print(msg, end='')
            author_guid_verdict_df = author_guid_binary_verdict_results_df.loc[
                author_guid_binary_verdict_results_df['author_guid'] == author_guid]

            author_sub_type = author_guid_verdict_df['author_sub_type'].values[0]
            num_of_replications = 1
            if self._replicate_authors_by == "num_of_posts":
                if author_sub_type == "fake_news_promoter":
                    fake_news_num_of_posts = author_guid_verdict_df['fake_news_num_of_posts'].values[0]
                    num_of_replications = fake_news_num_of_posts
                elif author_sub_type == "real_news_promoter":
                    fake_news_num_of_posts = author_guid_verdict_df['real_news_num_of_posts'].values[0]
                    num_of_replications = fake_news_num_of_posts
            else:
                num_of_distinct_claims = author_guid_verdict_df['num_of_distinct_claims'].values[0]
                num_of_replications = num_of_distinct_claims

            author_guid_df = training_df.loc[[author_guid]]

            replicated_training_df = replicated_training_df.append([author_guid_df] * num_of_replications,
                                                                   ignore_index=True)

        # shuffle the rows
        replicated_training_df = replicated_training_df.sample(frac=1).reset_index(drop=True)
        return replicated_training_df

    def _update_dataset_statistics(self, dataset_name, dataset_df):
        optional_class_1 = self._optional_classes[0]
        num_of_optional_class_1_actors = self._get_number_of_actors_by_dataset_and_optional_class(dataset_df,
                                                                                                  optional_class_1)

        optional_class_2 = self._optional_classes[1]
        num_of_optional_class_2_actors = self._get_number_of_actors_by_dataset_and_optional_class(dataset_df,
                                                                                                  optional_class_2)

        max_num_of_actors = max(num_of_optional_class_1_actors, num_of_optional_class_2_actors)
        sun_num_of_actors = num_of_optional_class_1_actors + num_of_optional_class_2_actors

        dataset_tuple = (dataset_name, num_of_optional_class_1_actors, num_of_optional_class_2_actors,
                         max_num_of_actors / float(sun_num_of_actors))
        self._dataset_authors_distribution_tuples.append(dataset_tuple)

    def _get_number_of_actors_by_dataset_and_optional_class(self, dataset_df, optional_class):
        dataset_df_optional_class = dataset_df[dataset_df[self._targeted_class_name] == optional_class]
        dataset_df_optional_class = dataset_df_optional_class.reset_index()
        num_of_optional_class_actors = len(dataset_df_optional_class.index)
        return num_of_optional_class_actors

    def predict_based_on_authors_distribution(self, test_claim_id, test_claim_post_author_connection,
                                              author_guid_author_statistics_df, k_fold_iteration_num,
                                              test_claim_index, test_class_series):
        self._classifier_type = "predict_based_authors_distribution"

        test_claim_author_guids_series = test_claim_post_author_connection['author_guid']
        test_claim_author_guids = test_claim_author_guids_series.tolist()
        test_claim_author_guids = list(set(test_claim_author_guids))

        fake_news_promoter_count = 0
        real_news_promoter_count = 0
        for test_claim_author_guid in test_claim_author_guids:
            test_author_guid_author_sub_type_df = author_guid_author_statistics_df[
                author_guid_author_statistics_df['author_guid'] == test_claim_author_guid]
            if not test_author_guid_author_sub_type_df.empty:
                label_series = test_author_guid_author_sub_type_df['author_sub_type']
                author_sub_type = label_series.values[0]
                if author_sub_type == "fake_news_promoter":
                    fake_news_promoter_count += 1
                elif author_sub_type == "real_news_promoter":
                    real_news_promoter_count += 1

        is_random_guess = False
        prediction = -1
        if fake_news_promoter_count > real_news_promoter_count:
            prediction = 1
        elif fake_news_promoter_count < real_news_promoter_count:
            prediction = 0
        else:
            is_random_guess = True
            random_prediction = random.randint(0, 1)
            prediction = random_prediction

        ground_truth_result = test_class_series[test_claim_index]
        test_claim_prediction_tuple = (test_claim_id, k_fold_iteration_num, fake_news_promoter_count,
                                       real_news_promoter_count, is_random_guess, prediction,
                                       ground_truth_result)
        self._test_claim_prediction_tuples.append(test_claim_prediction_tuple)

        return prediction

    def predict_based_on_authors_distribution_prob(self, test_claim_id,
                                                   test_claim_post_author_connection,
                                                   author_guid_author_statistics_df,
                                                   k_fold_iteration_num, test_claim_index,
                                                   test_class_series):
        test_claim_author_guids_series = test_claim_post_author_connection['author_guid']
        test_claim_author_guids = test_claim_author_guids_series.tolist()
        test_claim_author_guids = list(set(test_claim_author_guids))

        fake_news_promoter_count = 0
        real_news_promoter_count = 0
        for test_claim_author_guid in test_claim_author_guids:
            test_author_guid_author_sub_type_df = author_guid_author_statistics_df[
                author_guid_author_statistics_df['author_guid'] == test_claim_author_guid]
            if not test_author_guid_author_sub_type_df.empty:
                label_series = test_author_guid_author_sub_type_df['author_sub_type']
                author_sub_type = label_series.values[0]
                if author_sub_type == "fake_news_promoter":
                    fake_news_promoter_count += 1
                elif author_sub_type == "real_news_promoter":
                    real_news_promoter_count += 1

        is_random_guess = False
        overall = fake_news_promoter_count + real_news_promoter_count
        fake_news_promoter_prob = fake_news_promoter_count / overall
        real_news_promoter_prob = real_news_promoter_count / overall
        if fake_news_promoter_count > real_news_promoter_count:
            prediction = fake_news_promoter_prob
        elif fake_news_promoter_count < real_news_promoter_count:
            prediction = 1 - real_news_promoter_prob
        else:
            is_random_guess = True
            prediction = 0.5

        ground_truth_result = test_class_series[test_claim_index]
        test_claim_prediction_tuple = (test_claim_id, k_fold_iteration_num, fake_news_promoter_count,
                                       real_news_promoter_count, is_random_guess,
                                       prediction, ground_truth_result)
        self._test_claim_prediction_tuples.append(test_claim_prediction_tuple)

        return prediction

    def predict_based_posts_distribution(self, test_claim_id,
                                         test_claim_post_author_connection,
                                         author_guid_author_statistics_df,
                                         k_fold_iteration_num,
                                         verdict):

        test_claim_post_author_connection_author_sub_type = test_claim_post_author_connection.join(
            author_guid_author_statistics_df, lsuffix='_test_claim_post_author_connection',
            rsuffix='_author_guid_author_statistics_df')

        test_claim_post_author_connection_fake_news_promoters = test_claim_post_author_connection_author_sub_type[
            test_claim_post_author_connection_author_sub_type['author_sub_type'] == 'fake_news_promoter']
        test_claim_post_author_connection_fake_news_promoters = test_claim_post_author_connection_fake_news_promoters.reset_index()
        fake_news_promoters_num_of_posts = len(test_claim_post_author_connection_fake_news_promoters.index)

        test_claim_post_author_connection_real_news_promoters = test_claim_post_author_connection_author_sub_type[
            test_claim_post_author_connection_author_sub_type['author_sub_type'] == 'real_news_promoter']
        test_claim_post_author_connection_real_news_promoters = test_claim_post_author_connection_real_news_promoters.reset_index()
        real_news_promoters_num_of_posts = len(test_claim_post_author_connection_real_news_promoters.index)

        is_random_guess = False

        if fake_news_promoters_num_of_posts > real_news_promoters_num_of_posts:
            overall = fake_news_promoters_num_of_posts + real_news_promoters_num_of_posts
            fake_news_promoters_num_of_posts_prob = fake_news_promoters_num_of_posts / float(overall)
            confidence_prediction = fake_news_promoters_num_of_posts_prob

        elif fake_news_promoters_num_of_posts < real_news_promoters_num_of_posts:
            overall = fake_news_promoters_num_of_posts + real_news_promoters_num_of_posts
            real_news_promoters_num_of_posts_prob = real_news_promoters_num_of_posts / float(overall)
            confidence_prediction = 1 - real_news_promoters_num_of_posts_prob
        else:
            is_random_guess = True
            confidence_prediction = 0.5

        if confidence_prediction > 0.5:
            binary_prediction = 1
        elif confidence_prediction < 0.5:
            binary_prediction = 0
        else:
            binary_prediction = random.randint(0, 1)

        ground_truth_result = verdict
        test_claim_prediction_tuple = (test_claim_id, k_fold_iteration_num, fake_news_promoters_num_of_posts,
                                       real_news_promoters_num_of_posts, is_random_guess,
                                       confidence_prediction, binary_prediction, ground_truth_result)
        self._test_claim_prediction_tuples.append(test_claim_prediction_tuple)

        return confidence_prediction, binary_prediction

    def _calculate_performance(self, test_class, total_confidence_predictions, total_binary_predictions,
                               selected_features):
        try:
            auc_score = roc_auc_score(test_class, total_confidence_predictions)
        except:
            auc_score = -1
        accuracy = accuracy_score(test_class, total_binary_predictions)
        f1 = f1_score(test_class, total_binary_predictions)
        precision = precision_score(test_class, total_binary_predictions)
        recall = recall_score(test_class, total_binary_predictions)

        result_performance_tuple = (selected_features, auc_score, accuracy, f1, precision, recall)
        return result_performance_tuple

    def _build_training_claim_post_author_connections(self, training_df):
        training_claim_post_author_connection_df = pd.DataFrame()

        training_claim_ids_series = training_df['claim_id']
        training_claim_ids = training_claim_ids_series.tolist()
        training_claim_ids = list(set(training_claim_ids))
        for training_claim_id in training_claim_ids:
            claim_tweet_author_connection_by_claim_df = self._claim_post_author_connection_df[
                self._claim_post_author_connection_df['claim_id'] == training_claim_id]
            # claim_tweet_author_connection_df = claim_tweet_author_connection_df.reset_index()

            training_claim_post_author_connection_df = training_claim_post_author_connection_df.append(
                claim_tweet_author_connection_by_claim_df, ignore_index=True)
        return training_claim_post_author_connection_df

    def _label_authors_based_on_training_claims(self, training_claim_post_author_connection_df, iteration_num):
        # get training claims
        training_claims_authors_guids_series = training_claim_post_author_connection_df['author_guid']
        training_claims_author_guids = training_claims_authors_guids_series.tolist()
        training_claims_author_guids = list(set(training_claims_author_guids))

        self._author_guid_by_binary_verdict_tuples = []
        self._classifier_type = ""

        author_guid_statistics_tuples = []
        for i, training_author_guid in enumerate(training_claims_author_guids):
            # ##########################
            # # Need to remove just for testing
            # if i == 1000:
            #     break
            msg = "\r Loop on training authors for automatic lableing: {0}/{1}".format(i, len(
                training_claims_author_guids))
            print(msg, end="")

            training_claim_by_author_guid_df = training_claim_post_author_connection_df[
                training_claim_post_author_connection_df['author_guid'] == training_author_guid]

            training_claim_by_author_guid_df = training_claim_by_author_guid_df.reset_index()
            num_of_posts = len(training_claim_by_author_guid_df.index)

            # self._organize_author_guid_by_verdicts_by_statistics(author_guid, claim_by_author_guid_df)

            training_author_distinct_claims_df = training_claim_by_author_guid_df.groupby(
                ['author_guid', 'claim_id', 'verdict']).size().reset_index(name='counts')

            # author_claim_df_results = author_claim_df_results.append(distinct_claims_df, ignore_index=True)

            num_of_distinct_claims = len(training_author_distinct_claims_df.index)
            label = ''
            if num_of_distinct_claims >= self._min_num_of_distinct_claims_author_involved and \
                    num_of_posts >= self._min_num_of_published_posts:

                # claim_by_author_guid_df = self._convert_df_to_binary_problem(claim_by_author_guid_df)

                author_guid_by_verict_df = training_claim_by_author_guid_df[
                    training_claim_by_author_guid_df['verdict'] == 1]
                author_guid_by_verict_df = author_guid_by_verict_df.reset_index()
                num_of_fake_news_posts = len(author_guid_by_verict_df.index)

                author_guid_by_verict_df = training_claim_by_author_guid_df[
                    training_claim_by_author_guid_df['verdict'] == 0]
                author_guid_by_verict_df = author_guid_by_verict_df.reset_index()
                num_of_real_news_posts = len(author_guid_by_verict_df.index)

                fake_news_posts_distribution = num_of_fake_news_posts / float(num_of_posts)
                real_news_posts_distribution = num_of_real_news_posts / float(num_of_posts)

                if fake_news_posts_distribution >= self._threshold:
                    label = 'fake_news_promoter'
                    # author.author_sub_type = self._fake_news_promoter_label
                elif real_news_posts_distribution >= self._threshold:
                    label = 'real_news_promoter'
                    # author.author_sub_type = self._real_news_promoter_label
                author_guid_statistics_tuple = (
                    training_author_guid, iteration_num, num_of_distinct_claims, num_of_posts,
                    num_of_fake_news_posts, num_of_real_news_posts, label)
                author_guid_statistics_tuples.append(author_guid_statistics_tuple)

            else:
                author_guid_statistics_tuple = (training_author_guid, iteration_num, num_of_distinct_claims,
                                                num_of_posts, 0, 0, label)
                author_guid_statistics_tuples.append(author_guid_statistics_tuple)

        author_guid_statistics_df = pd.DataFrame(author_guid_statistics_tuples,
                                                 columns=['author_guid', 'k_fold_cross_validation_iteration_num',
                                                          'num_of_distinct_claims', 'num_of_posts',
                                                          'num_of_fake_news_posts', 'num_of_real_news_posts',
                                                          'author_sub_type'])
        return author_guid_statistics_df

    def _predict_test_claims(self, test_df, author_guid_author_sub_type_df,
                             k_fold_iteration_num):
        test_claim_ids_series = test_df['claim_id']
        test_claim_ids = test_claim_ids_series.tolist()

        verdict_series = test_df['verdict']
        verdicts = verdict_series.tolist()

        confidence_predictions = []
        binary_predictions = []
        for i, test_claim_id in enumerate(test_claim_ids):
            msg = "\r Predict on test claims: {0}/{1}".format(i, len(test_claim_ids))
            print(msg, end="")

            verdict = verdicts[i]
            test_claim_post_author_connection_df = self._claim_post_author_connection_df[
                self._claim_post_author_connection_df['claim_id'] == test_claim_id]

            confidence_prediction, binary_prediction = getattr(self, self._prediction_function)(test_claim_id,
                                                                                                test_claim_post_author_connection_df,
                                                                                                author_guid_author_sub_type_df,
                                                                                                k_fold_iteration_num,
                                                                                                verdict)
            confidence_predictions.append(confidence_prediction)
            binary_predictions.append(binary_prediction)
        return confidence_predictions, binary_predictions

    def _convert_verdict_to_binary_problem(self, df):
        for general_label, original_labels in self._fake_and_real_news_labels_dict.items():
            for original_label in original_labels:
                df.replace(to_replace=original_label, value=general_label, inplace=True)

        for targeted_class, num in self._targeted_class_dict.items():
            df.replace(to_replace=targeted_class, value=num, inplace=True)

    def _create_claim_verdict_df(self, claims):
        claim_tuples = []
        for i, claim in enumerate(claims):
            claim_id = claim.claim_id
            verdict = claim.verdict
            claim_tuple = (i + 1, claim_id, verdict)

            claim_tuples.append(claim_tuple)

        claim_num_verdict_df = pd.DataFrame(claim_tuples, columns=['claim_num', 'claim_id', 'verdict'])
        return claim_num_verdict_df
