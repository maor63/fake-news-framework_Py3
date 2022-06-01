# Created by Aviad at 27/06/2017
from preprocessing_tools.abstract_controller import AbstractController
from commons.data_frame_creator import DataFrameCreator
from commons.consts import Classifiers, PerformanceMeasures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn import tree
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,confusion_matrix, precision_recall_fscore_support
from .results_container import ResultsContainer
import os
import joblib
from sklearn.model_selection import StratifiedKFold
import numpy as np

class Clickbait_Challenge_Evaluator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

        self._set_type_affiliation = 'set_affiliation'
        self._targeted_class_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_dict")
        self._targeted_class_field_names = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_names")
        self._classifier_type_names = self._config_parser.eval(self.__class__.__name__, "classifier_type_names")
        self._num_of_features = self._config_parser.eval(self.__class__.__name__, "num_of_features")
        self._indentifier_field_name = self._config_parser.get(self.__class__.__name__, "indentifier_field_name")
        self._replace_missing_values = self._config_parser.get(self.__class__.__name__, "replace_missing_values")
        self._selected_features = self._config_parser.eval(self.__class__.__name__, "selected_features")

        self._feature_names_to_remove = self._config_parser.eval(self.__class__.__name__, "feature_names_to_remove")
        self._order_of_results_dictionary = self._config_parser.eval(self.__class__.__name__, "order_of_results_dictionary")
        self._results_file_name = self._config_parser.get(self.__class__.__name__, "results_file_name")
        self._results_table_file_name = self._config_parser.get(self.__class__.__name__, "results_table_file_name")
        self._path = self._config_parser.get(self.__class__.__name__, "path")
        self._column_names_for_results_table = self._config_parser.eval(self.__class__.__name__, "column_names_for_results_table")
        self._full_path_model_directory = self._config_parser.get(self.__class__.__name__, "full_path_model_directory")
        self._is_divide_to_training_and_test_sets_by_field_name_then_train_and_evaluate = self._config_parser.eval(self.__class__.__name__, "is_divide_to_training_and_test_sets_by_field_name_then_train_and_evaluate")
        self._is_divide_to_training_and_test_sets_by_k_fold_cross_validation_then_train_and_evaluate = self._config_parser.eval(self.__class__.__name__, "is_divide_to_training_and_test_sets_by_k_fold_cross_validation_then_train_and_evaluate")
        self._k_for_fold = self._config_parser.eval(self.__class__.__name__, "k_for_fold")


        self._training_label = "training"
        self._test_label = "test"

    def execute(self, window_start=None):

        self._reversed_targeted_class_dict = {num: targeted_class for targeted_class, num in self._targeted_class_dict.items()}

        author_features_dataframe = self._get_author_features_dataframe()

        full_path = self._path + "author_features_dataframe.csv"

        author_features_dataframe.to_csv(full_path, index=False)


        for targeted_class, num in self._targeted_class_dict.items():
            author_features_dataframe = author_features_dataframe.replace(to_replace=targeted_class, value=num)

        self._results_dict = self._create_results_dictionary()
        if self._is_divide_to_training_and_test_sets_by_field_name_then_train_and_evaluate:

            self._divide_to_training_and_test_sets_by_field_name_then_train_and_evaluate(author_features_dataframe)

        elif self._is_divide_to_training_and_test_sets_by_k_fold_cross_validation_then_train_and_evaluate:
             for targeted_class_field_name in self._targeted_class_field_names:
                labeled_features_dataframe = author_features_dataframe.loc[author_features_dataframe[targeted_class_field_name].notnull()]
                original_labeled_features_dataframe, targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(labeled_features_dataframe)

                for classifier_type_name in self._classifier_type_names:
                    for num_of_features in self._num_of_features:
                        labeled_features_dataframe = original_labeled_features_dataframe.copy()
                        targeted_dataframe, k_best_features = self._reduce_dimensions_by_num_of_features(labeled_features_dataframe, targeted_class_series, num_of_features)

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)

                        if selected_classifier is not None:

                            k_folds, valid_k = self._select_valid_k(targeted_class_series)

                            print(("Valid k is: " + str(valid_k)))
                            i = 0
                            for train_indexes, test_indexes in k_folds:
                                i += 1
                                print(("i = " + str(i)))

                                training_set_dataframe, test_set_dataframe, training_targeted_class_series, test_targeted_class_series = self._create_train_and_test_dataframes_and_classes(targeted_dataframe, train_indexes, test_indexes, targeted_class_series)

                                self._select_classifier_train_and_evaluate_performance(targeted_class_field_name,
                                                                                       num_of_features,
                                                                                       classifier_type_name,
                                                                                       training_set_dataframe,
                                                                                       training_targeted_class_series,
                                                                                       k_best_features,
                                                                                       test_set_dataframe,
                                                                                       test_targeted_class_series)

                self._result_container.calculate_average_performances(valid_k)
                self._result_container.write_results_as_table()

                best_classifier_dict = self._find_best_classifier_train_with_all_labeled_dataset_and_save(
                    author_features_dataframe)

                self._use_classifier_for_unlabeled_prediction(best_classifier_dict, author_features_dataframe)


    def _save_trained_model(self, selected_classifier, targeted_class_field_name,  classifier_type_name, num_of_features, reduced_dataframe_column_names):
        if not os.path.exists(self._full_path_model_directory):
            os.makedirs(self._full_path_model_directory)

        # save model
        full_model_file_path = self._full_path_model_directory + "trained_classifier_" + targeted_class_field_name + "_" + \
                               classifier_type_name + "_" + str(
            num_of_features) + "_features.pkl"
        joblib.dump(selected_classifier, full_model_file_path)

        # save features
        model_features_file_path = self._full_path_model_directory + "trained_classifier_" + targeted_class_field_name + "_" + classifier_type_name + "_" + str(
            num_of_features) + "_selected_features.pkl"
        joblib.dump(reduced_dataframe_column_names, model_features_file_path)


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

    def _prepare_dataframe_for_learning(self, dataframe):
        dataframe.reset_index(drop=True, inplace=True)

        # Replace None in 0 for later calculation
        if self._replace_missing_values == 'zero':
            dataframe = dataframe.fillna(0)
        elif self._replace_missing_values == 'mean':
            dataframe = dataframe.fillna(dataframe.mean())

        #dataframe = replace_nominal_class_to_numeric(dataframe, self._optional_classes)

        indentifier_series = dataframe.pop(self._indentifier_field_name)
        targeted_class_series = dataframe.pop(self._targeted_class_field_names[0])

        # Remove unnecessary features
        dataframe = self._remove_features(self._feature_names_to_remove, dataframe)

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

        return targeted_dataframe, targeted_class_series, indentifier_series

    def _reduce_dimensions_by_num_of_features(self, labeled_author_features_dataframe, targeted_class_series,
                                              num_of_features):
        print(("Create dataframe with the {0} best features".format(num_of_features)))

        return self._find_k_best_and_reduce_dimensions(num_of_features, labeled_author_features_dataframe,
                                                       targeted_class_series)

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

    def _create_dataframe_by_k_best_features(self, k_best_features, reduced_dataframe_column_names):
        reduced_dataframe = pd.DataFrame(k_best_features, columns=reduced_dataframe_column_names)

        return reduced_dataframe


    def _get_author_features_dataframe(self):
        data_frame_creator = DataFrameCreator(self._db)
        data_frame_creator.create_author_features_data_frame()
        author_features_dataframe = data_frame_creator.get_author_features_data_frame()
        return author_features_dataframe

    def _remove_features(self, features_to_remove, dataframe):
        '''
        This function is responsible to remove features.
        :param dataframe:
        :return:dataframe without removed columns
        '''
        # print("Start remove_unnecessary_features")
        # Remove unnecessary features
        dataframe_columns = list(dataframe.columns.values)
        for unnecessary_feature in features_to_remove:
            if unnecessary_feature in dataframe_columns:
                dataframe.pop(unnecessary_feature)

        return dataframe

    def _create_results_dictionary(self):

        results_dictionary_compenents = []
        for list_name in self._order_of_results_dictionary:
            elements = getattr(self, "_"+ list_name)
            results_dictionary_compenents.append(elements)

        self._result_container = ResultsContainer(self._path, self._results_table_file_name, self._column_names_for_results_table, results_dictionary_compenents)
        self._results_dict = self._result_container.get_results()
        return self._results_dict

    def _calculate_performance_measures(self, test_targeted_class_series, test_set_predictions,
                                        targeted_class_field_name, classifier_type_name, num_of_features, k_best_features):

        self._result_container.set_result(k_best_features,PerformanceMeasures.SELECTED_FEATURES,targeted_class_field_name,
                                          classifier_type_name, num_of_features)

        auc_score = roc_auc_score(test_targeted_class_series, test_set_predictions)
        self._result_container.set_result(auc_score, PerformanceMeasures.AUC, targeted_class_field_name,
                                          classifier_type_name, num_of_features)
        prediction = precision_score(test_targeted_class_series, test_set_predictions)
        self._result_container.set_result(prediction, PerformanceMeasures.PRECISION,
                                          targeted_class_field_name,
                                          classifier_type_name, num_of_features)
        recall = recall_score(test_targeted_class_series, test_set_predictions)
        self._result_container.set_result(recall, PerformanceMeasures.RECALL,
                                          targeted_class_field_name,
                                          classifier_type_name, num_of_features)
        accuracy = accuracy_score(test_targeted_class_series, test_set_predictions)
        self._result_container.set_result(accuracy, PerformanceMeasures.ACCURACY,
                                          targeted_class_field_name,
                                          classifier_type_name, num_of_features)

        confusion_matrix_score = confusion_matrix(test_targeted_class_series, test_set_predictions)
        print(("confusion_matrix is: " + str(confusion_matrix_score)))
        self._result_container.set_result(confusion_matrix_score, PerformanceMeasures.CONFUSION_MATRIX,
                                          targeted_class_field_name,
                                          classifier_type_name, num_of_features)

        num_of_correct_instances, num_of_incorrect_instances = self._result_container.calculate_correctly_and_not_correctly_instances(
            confusion_matrix_score)

        self._result_container.set_result(num_of_correct_instances,
                                          PerformanceMeasures.CORRECTLY_CLASSIFIED,
                                          targeted_class_field_name,
                                          classifier_type_name, num_of_features)

        self._result_container.set_result(num_of_incorrect_instances,
                                          PerformanceMeasures.INCORRECTLY_CLASSIFIED,
                                          targeted_class_field_name,
                                          classifier_type_name, num_of_features)

    def _calculate_features_to_remove(self, best_column_names, training_columns):
        best_combination_set = set(best_column_names)
        training_columns_set = set(training_columns)
        features_to_remove_set = training_columns_set - best_combination_set
        features_to_remove = list(features_to_remove_set)
        return features_to_remove

    def _select_classifier_train_and_evaluate_performance(self, targeted_class_field_name, num_of_features, classifier_type_name,
                                                          training_set_dataframe,training_targeted_class_series,
                                                          k_best_features,test_set_dataframe, test_targeted_class_series):
        selected_classifier = self._select_classifier_by_type(classifier_type_name)
        if selected_classifier is not None:
            selected_classifier.fit(training_set_dataframe, training_targeted_class_series)

            if classifier_type_name == Classifiers.XGBoost:
                for column_name in k_best_features:
                    test_set_dataframe[column_name] = test_set_dataframe[column_name].astype(float)

            test_set_predictions = selected_classifier.predict(test_set_dataframe)

            self._calculate_performance_measures(test_targeted_class_series, test_set_predictions,
                                                 targeted_class_field_name,
                                                 classifier_type_name, num_of_features, k_best_features)

    def _find_best_classifier_train_with_all_labeled_dataset_and_save(self, author_features_dataframe):
        best_classifier_dict = {}
        for targeted_class_field_name in self._targeted_class_field_names:
            selected_combination, best_features = self._result_container.find_max_average_auc_classifier()
            best_classifier_name = selected_combination[1]
            num_of_features = selected_combination[2]

            full_dataframe, full_dataframe_targeted_class_series, full_dataframe_identifier_series = self._prepare_dataframe_for_learning(
                author_features_dataframe)

            full_dataframe_features = list(full_dataframe.columns.values)
            features_to_remove = self._calculate_features_to_remove(best_features, full_dataframe_features)
            reduced_feature_full_dataframe = self._remove_features(features_to_remove, full_dataframe)

            best_classifier = self._select_classifier_by_type(best_classifier_name)
            if best_classifier is not None:
                best_classifier.fit(reduced_feature_full_dataframe, full_dataframe_targeted_class_series)

                self._save_trained_model(best_classifier, targeted_class_field_name, best_classifier_name,num_of_features, best_features)

                best_classifier_dict[targeted_class_field_name] = (best_classifier_name, num_of_features, best_classifier)
                #For example {'author_type' : ('RandomForest', 10, CLASSIFIER object)
        return best_classifier_dict

    def _divide_to_training_and_test_sets_by_field_name_then_train_and_evaluate(self, author_features_dataframe):
        # retrieve all training set
        training_set_dataframe = author_features_dataframe.loc[
            author_features_dataframe[self._set_type_affiliation] == self._training_label]
        training_set_dataframe, training_targeted_class_series, training_identifier_series = self._prepare_dataframe_for_learning(
            training_set_dataframe)
        original_training_set_dataframe = training_set_dataframe.copy()

        test_set_dataframe = author_features_dataframe.loc[
            author_features_dataframe[self._set_type_affiliation] == self._test_label]
        test_set_dataframe, test_targeted_class_series, test_identifier_series = self._prepare_dataframe_for_learning(
            test_set_dataframe)
        original_test_set_dataframe = test_set_dataframe.copy()

        #self._results_dict = self._create_results_dictionary()

        for targeted_class_field_name in self._targeted_class_field_names:
            for classifier_type_name in self._classifier_type_names:
                for num_of_features in self._num_of_features:
                    print(("{0}-{1}".format(classifier_type_name, num_of_features)))

                    training_set_dataframe = original_training_set_dataframe.copy()
                    test_set_dataframe = original_test_set_dataframe.copy()

                    training_columns = list(original_training_set_dataframe.columns.values)
                    training_set_dataframe, k_best_features = self._reduce_dimensions_by_num_of_features(
                        training_set_dataframe, training_targeted_class_series, num_of_features)

                    features_to_remove = self._calculate_features_to_remove(k_best_features, training_columns)

                    test_set_dataframe = self._remove_features(features_to_remove, test_set_dataframe)


                    self._select_classifier_train_and_evaluate_performance(targeted_class_field_name, num_of_features, classifier_type_name, training_set_dataframe,
                                                                           training_targeted_class_series,
                                                                           k_best_features, test_set_dataframe,
                                                                           test_targeted_class_series)

        self._result_container.write_results_as_table()

        best_classifier_dict = self._find_best_classifier_train_with_all_labeled_dataset_and_save(author_features_dataframe)

        self._use_classifier_for_unlabeled_prediction(best_classifier_dict, author_features_dataframe)



    def _predict_classifier(self, selected_classifier, unlabeled_author_dataframe):
        predictions = selected_classifier.predict(unlabeled_author_dataframe)
        predictions_series = pd.Series(predictions)

        predictions_series = self._replace_predictions_class_from_int_to_string(predictions_series)

        predictions_proba = selected_classifier.predict_proba(unlabeled_author_dataframe)

        optional_classes = list(self._targeted_class_dict.keys())
        num_of_classes = len(optional_classes)
        if num_of_classes == 2:
            predictions_proba_series = pd.Series(predictions_proba[:, 1])
        elif num_of_classes > 2:
            predictions_proba_ndarray = np.array(predictions_proba)
            max_predictions_proba = predictions_proba_ndarray.max(axis=1)
            predictions_proba_series = pd.Series(max_predictions_proba)
        return predictions_series, predictions_proba_series

    def _replace_predictions_class_from_int_to_string(self, predictions_series):
        predictions_series = self._replace_numeric_class_to_nominal(predictions_series)
        return predictions_series

    def _replace_numeric_class_to_nominal(self, dataframe):
        for targeted_class_name, num in self._targeted_class_dict:
            dataframe = dataframe.replace(to_replace=num, value=targeted_class_name)
        return dataframe


    def _select_valid_k(self, targeted_class_series):
        valid_k = self._retreive_valid_k(self._k_for_fold, targeted_class_series)
        k_folds = StratifiedKFold(targeted_class_series, valid_k)
        return k_folds, valid_k

    def _retreive_valid_k(self, k, author_type_class_series):
        series_length = len(author_type_class_series)
        if series_length < k:
            return series_length
        else:
            return k

    def _create_train_and_test_dataframes_and_classes(self, targeted_dataframe, train_indexes, test_indexes, targeted_class_series):
        train_set_dataframe = targeted_dataframe.loc[train_indexes.tolist()]
        test_set_dataframe = targeted_dataframe.loc[test_indexes.tolist()]
        train_class = targeted_class_series[train_indexes]
        test_class = targeted_class_series[test_indexes]
        return train_set_dataframe, test_set_dataframe, train_class, test_class

    def _use_classifier_for_unlabeled_prediction(self, best_classifier_dict, author_features_dataframe):
        for targeted_class_field_name in self._targeted_class_field_names:
            unlabeled_features_dataframe = author_features_dataframe.loc[author_features_dataframe[targeted_class_field_name].isnull()]
            if not unlabeled_features_dataframe.empty:

                unlabeled_features_dataframe, unlabeled_targeted_class_series, unlabeled_index_field_series = self._prepare_dataframe_for_learning(unlabeled_features_dataframe)

                tuple = best_classifier_dict[targeted_class_field_name]
                best_classifier_name = tuple[0]
                num_of_features = tuple[1]
                best_classifier = tuple[2]

                unlabeled_author_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                    unlabeled_features_dataframe, unlabeled_targeted_class_series, num_of_features)

                predictions_series, predictions_proba_series = self._predict_classifier(best_classifier, unlabeled_author_dataframe)

                self._write_predictions_into_file(best_classifier_name, num_of_features,
                                                  unlabeled_index_field_series, predictions_series,
                                                  predictions_proba_series)


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

            table_name = targeted_class_field_name + "unlabeled_predictions"
            self._db.drop_unlabeled_predictions(table_name)

            engine = self._db.engine
            unlabeled_dataframe_with_prediction.to_sql(name=table_name, con=engine)


