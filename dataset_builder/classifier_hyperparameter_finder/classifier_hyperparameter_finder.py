

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from commons.method_executor import Method_Executor
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
import itertools
import time
import xgboost as xgb
from sklearn import tree, svm
import numpy as np
import os


__author__ = "Aviad Elyashar"

class ClassifierHyperparameterFinder(Method_Executor):

    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_path = self._config_parser.eval(self.__class__.__name__, "input_path")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._dataset_csv_file_name = self._config_parser.eval(self.__class__.__name__, "dataset_csv_file_name")
        self._sort_by_date = self._config_parser.eval(self.__class__.__name__, "sort_by_date")
        self._date_field_name = self._config_parser.eval(self.__class__.__name__, "date_field_name")
        self._remove_features = self._config_parser.eval(self.__class__.__name__, "remove_features")
        self._targeted_class_field_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_name")
        self._training_set_num_of_records = self._config_parser.eval(self.__class__.__name__, "training_set_num_of_records")
        self._test_set_num_of_records = self._config_parser.eval(self.__class__.__name__, "test_set_num_of_records")
        self._validation_set_num_of_records = self._config_parser.eval(self.__class__.__name__, "validation_set_num_of_records")
        self._num_of_features = self._config_parser.eval(self.__class__.__name__, "num_of_features")
        self._random_forest_estimators = self._config_parser.eval(self.__class__.__name__, "random_forest_estimators")
        self._random_forest_max_depth = self._config_parser.eval(self.__class__.__name__, "random_forest_max_depth")
        self._random_forest_min_samples_split = self._config_parser.eval(self.__class__.__name__, "random_forest_min_samples_split")
        self._random_forest_min_samples_leaf = self._config_parser.eval(self.__class__.__name__, "random_forest_min_samples_leaf")
        self._random_forest_max_leaf_nodes = self._config_parser.eval(self.__class__.__name__, "random_forest_max_leaf_nodes")

        self._xgboost_estimators = self._config_parser.eval(self.__class__.__name__, "xgboost_estimators")
        self._xgboost_max_depth = self._config_parser.eval(self.__class__.__name__, "xgboost_max_depth")
        self._xgboost_learning_rate = self._config_parser.eval(self.__class__.__name__, "xgboost_learning_rate")

        self._adaboost_estimators = self._config_parser.eval(self.__class__.__name__, "adaboost_estimators")
        self._adaboost_learning_rate = self._config_parser.eval(self.__class__.__name__, "adaboost_learning_rate")

        self._decision_tree_max_depth = self._config_parser.eval(self.__class__.__name__, "decision_tree_max_depth")
        self._decision_tree_min_samples_split = self._config_parser.eval(self.__class__.__name__, "decision_tree_min_samples_split")
        self._decision_tree_min_samples_leaf = self._config_parser.eval(self.__class__.__name__, "decision_tree_min_samples_leaf")
        self._decision_tree_max_leaf_nodes = self._config_parser.eval(self.__class__.__name__, "decision_tree_max_leaf_nodes")


        self._svm_c = self._config_parser.eval(self.__class__.__name__, "svm_c")
        self._svm_degree = self._config_parser.eval(self.__class__.__name__, "svm_degree")



        # for run_selected_classiifier_on_training_and_test_sets
        self._iterations = self._config_parser.eval(self.__class__.__name__, "iterations")
        self._selected_num_of_features = self._config_parser.eval(self.__class__.__name__, "selected_num_of_features")
        self._selected_estimators = self._config_parser.eval(self.__class__.__name__, "selected_estimators")
        self._selected_max_depth = self._config_parser.eval(self.__class__.__name__, "selected_max_depth")
        self._selected_min_samples_split = self._config_parser.eval(self.__class__.__name__, "selected_min_samples_split")
        self._selected_min_samples_leaf = self._config_parser.eval(self.__class__.__name__, "selected_min_samples_leaf")
        self._selected_max_leaf_nodes = self._config_parser.eval(self.__class__.__name__, "selected_max_leaf_nodes")
        self._selected_top_features = self._config_parser.eval(self.__class__.__name__, "selected_top_features")
        self._num_to_class_dict = self._config_parser.eval(self.__class__.__name__, "num_to_class_dict")

    def setUp(self):
        if not os.path.isdir(self._output_path):
            os.makedirs(self._output_path)
    ##
    # This function was asked by Asaf.
    # He asked to choose the train and the validation randomly.
    # He asked also TPR and FPR
    ##
    def find_random_forest_best_parameters_random_train_validation_and_test_sets(self):
        claim_features_df = self._read_csv_file_and_preprocess_dataset()
        X_original_train, X_test = self._split_vals(claim_features_df, self._training_set_num_of_records)

        X_train = X_original_train.sample(n=200)
        # find elements in df1 that are not in df2
        X_validation = X_original_train[~(X_original_train['author_guid'].isin(X_train['author_guid']))].reset_index(drop=True)

        y_train = X_train[self._targeted_class_field_name]
        y_validation = X_validation[self._targeted_class_field_name]

        X_train = self._remove_selected_features(X_train)
        X_validation = self._remove_selected_features(X_validation)

        X_train.pop(self._targeted_class_field_name)
        X_validation.pop(self._targeted_class_field_name)

        X_train.pop(self._date_field_name)
        X_validation.pop(self._date_field_name)


        training_set_size, validation_set_size, train_num_of_false_records, train_num_of_true_records, \
        validation_num_of_false_records, \
        validation_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_train,
                                                                                              y_validation)

        important_features_df = self._create_default_random_forest_classifier_and_calculate_feature_importance(X_train, y_train)

        combinations = self._add_random_forest_parameters_and_create_combinations()

        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_random_forest_classifier_train_and_calculate_performance(combination,
                            important_features_df, X_train, X_validation, y_train, y_validation, training_set_size,
                            validation_set_size,
                           train_num_of_false_records,
                           train_num_of_true_records,
                           validation_num_of_false_records,
                           validation_num_of_true_records,
                           iteration)

                performance_results.append(performance_with_parameters)



        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Validation',
                                                                    'Accuracy Validation', 'F1 validation',
                                                                    'Precision Validation', 'Recall Validation',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Validation Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Validation)', '#True(Validation)',
                                                                    'Estimators', 'Max Depth', 'Min Samples Split',
                                                                    'Min Samples Leaf', 'Max Leaf Nodes', 'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "RandomForest_Results_on_Training_and_Validation_sets_Random_Validation_set.csv")



    def find_svm_best_parameters_train_and_test_sets_no_Validation(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()

        training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
        test_num_of_false_records, \
        test_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_original_train,
                                                                                                    y_test)

        important_features_df = self._create_default_svm_classifier_and_calculate_feature_importance(X_original_train,
                                                                                                               y_original_train)


        combinations = self._add_svm_parameters_and_create_combinations()
        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_svm_classifier_train_and_calculate_performance(combination,
                                                                                                    important_features_df,
                                                                                                           X_original_train,
                                                                                                           X_test,
                                                                                                           y_original_train,
                                                                                                           y_test,
                                                                                                           training_set_size,
                                                                                                           test_set_size,
                                                                                                           train_num_of_false_records,
                                                                                                           train_num_of_true_records,
                                                                                                           test_num_of_false_records,
                                                                                                           test_num_of_true_records,
                                                                                                           iteration)

                performance_results.append(performance_with_parameters)

        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Test',
                                                                    'Accuracy Test', 'F1 Test',
                                                                    'Precision Test', 'Recall Test',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Test Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Test)', '#True(Test)',
                                                                    'SVM_C', 'SVM Degree',
                                                                    'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "SVM_Results_on_Training_and_Test_sets_No_Validation.csv")


    def find_decision_tree_best_parameters_train_and_test_sets_no_Validation(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()

        training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
        test_num_of_false_records, \
        test_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_original_train,
                                                                                                    y_test)

        important_features_df = self._create_default_decision_tree_classifier_and_calculate_feature_importance(X_original_train,
                                                                                                               y_original_train)


        combinations = self._add_decision_tree_parameters_and_create_combinations()
        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_decision_tree_classifier_train_and_calculate_performance(combination,
                                                                                                    important_features_df,
                                                                                                           X_original_train,
                                                                                                           X_test,
                                                                                                           y_original_train,
                                                                                                           y_test,
                                                                                                           training_set_size,
                                                                                                           test_set_size,
                                                                                                           train_num_of_false_records,
                                                                                                           train_num_of_true_records,
                                                                                                           test_num_of_false_records,
                                                                                                           test_num_of_true_records,
                                                                                                           iteration)

                performance_results.append(performance_with_parameters)

        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Test',
                                                                    'Accuracy Test', 'F1 Test',
                                                                    'Precision Test', 'Recall Test',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Test Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Test)', '#True(Test)',
                                                                    'Max Depth', 'Min Samples Split',
                                                                    'Min Samples Leaf',
                                                                    'Max Leaf Nodes',
                                                                    'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "Decision_Tree_Results_on_Training_and_Test_sets_No_Validation.csv")



    def find_adaboost_best_parameters_train_and_test_sets_no_Validation(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()

        training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
        test_num_of_false_records, \
        test_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_original_train,
                                                                                                    y_test)

        important_features_df = self._create_default_adaboost_classifier_and_calculate_feature_importance(X_original_train,
                                                                                                               y_original_train)


        combinations = self._add_adaboost_parameters_and_create_combinations()
        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_adaboost_classifier_train_and_calculate_performance(combination,
                                                                                                    important_features_df,
                                                                                                           X_original_train,
                                                                                                           X_test,
                                                                                                           y_original_train,
                                                                                                           y_test,
                                                                                                     training_set_size,
                                                                                                     test_set_size,
                                                                                                           train_num_of_false_records,
                                                                                                           train_num_of_true_records,
                                                                                                           test_num_of_false_records,
                                                                                                           test_num_of_true_records,
                                                                                                           iteration)

                performance_results.append(performance_with_parameters)

        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Test',
                                                                    'Accuracy Test', 'F1 Test',
                                                                    'Precision Test', 'Recall Test',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Test Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Test)', '#True(Test)',
                                                                    'Estimators', 'Learning Rate',
                                                                    'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "Adaboost_Results_on_Training_and_Test_sets_No_Validation.csv")



    def find_xgboost_best_parameters_train_and_test_sets_no_Validation(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()

        training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
        test_num_of_false_records, \
        test_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_original_train,
                                                                                                    y_test)

        important_features_df = self._create_default_xgboost_classifier_and_calculate_feature_importance(X_original_train,
                                                                                                               y_original_train)

        X_test = self._convert_df_columns_to_float(X_test)


        combinations = self._add_xgboost_parameters_and_create_combinations()
        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_xgboost_classifier_train_and_calculate_performance(combination,
                                                                                                    important_features_df,
                                                                                                           X_original_train,
                                                                                                           X_test,
                                                                                                           y_original_train,
                                                                                                           y_test,
                                                                                                     training_set_size,
                                                                                                     test_set_size,
                                                                                                           train_num_of_false_records,
                                                                                                           train_num_of_true_records,
                                                                                                           test_num_of_false_records,
                                                                                                           test_num_of_true_records,
                                                                                                           iteration)

                performance_results.append(performance_with_parameters)

        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Test',
                                                                    'Accuracy Test', 'F1 Test',
                                                                    'Precision Test', 'Recall Test',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Test Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Test)', '#True(Test)',
                                                                    'Estimators', 'Max Depth', 'Learning Rate',
                                                                    'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "XGBoost_Results_on_Training_and_Test_sets_No_Validation.csv")



    def find_random_forest_best_parameters_train_validation_and_test_sets(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()

        X_train, X_validation, y_train, y_validation = self._create_training_and_validation_sets(X_original_train,
                                                                                                 y_original_train)

        training_set_size, validation_set_size, train_num_of_false_records, train_num_of_true_records, \
        validation_num_of_false_records, \
        validation_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_train,
                                                                                              y_validation)

        important_features_df = self._create_default_random_forest_classifier_and_calculate_feature_importance(X_train, y_train)

        combinations = self._add_random_forest_parameters_and_create_combinations()

        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_random_forest_classifier_train_and_calculate_performance(combination,
                            important_features_df, X_train, X_validation, y_train, y_validation, training_set_size,
                            validation_set_size,
                           train_num_of_false_records,
                           train_num_of_true_records,
                           validation_num_of_false_records,
                           validation_num_of_true_records,
                           iteration)

                performance_results.append(performance_with_parameters)



        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Validation',
                                                                    'Accuracy Validation', 'F1 validation',
                                                                    'Precision Validation', 'Recall Validation',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Validation Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Validation)', '#True(Validation)',
                                                                    'Estimators', 'Max Depth', 'Min Samples Split',
                                                                    'Min Samples Leaf', 'Max Leaf Nodes', 'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "RandomForest_Results_on_Training_and_Validation_sets.csv")


    def find_random_forest_best_parameters_train_and_test_sets_no_Validation(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()

        training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
        test_num_of_false_records, \
        test_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_original_train,
                                                                                                    y_test)

        important_features_df = self._create_default_random_forest_classifier_and_calculate_feature_importance(X_original_train,
                                                                                                               y_original_train)

        combinations = self._add_random_forest_parameters_and_create_combinations()
        performance_results = []
        for i, combination in enumerate(combinations):
            for iteration in range(1, self._iterations + 1):
                msg = "\r Combination: {0} / {1}, iteration = {2}".format(i, len(combinations), iteration)
                print(msg, end="")

                performance_with_parameters = \
                    self._read_combination_create_random_forest_classifier_train_and_calculate_performance(combination,
                                                                                                           important_features_df,
                                                                                                           X_original_train,
                                                                                                           X_test,
                                                                                                           y_original_train,
                                                                                                           y_test,
                                                                                                           training_set_size,
                                                                                                           test_set_size,
                                                                                                           train_num_of_false_records,
                                                                                                           train_num_of_true_records,
                                                                                                           test_num_of_false_records,
                                                                                                           test_num_of_true_records,
                                                                                                           iteration)



                performance_results.append(performance_with_parameters)

        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Test',
                                                                    'Accuracy Test', 'F1 Test',
                                                                    'Precision Test', 'Recall Test',
                                                                    'Classifier', 'Num of Features',
                                                                    'Training Set Size',
                                                                    'Test Set Size', '#False(Training)',
                                                                    '#True(Training)',
                                                                    '#False(Test)', '#True(Test)',
                                                                    'Estimators', 'Max Depth', 'Min Samples Split',
                                                                    'Min Samples Leaf', 'Max Leaf Nodes', 'Iteration',
                                                                    'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "RandomForest_Results_on_Training_and_Test_sets_No_Validation.csv")




    def run_selected_classiifier_on_training_and_test_sets(self):
        X_original_train, X_test, y_original_train, y_test = self._read_csv_file_and_divide_to_train_and_test_sets()


        training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
        test_num_of_false_records, \
        test_num_of_true_records = self._get_set_sizes_and_true_and_false_records_in_each_set(y_original_train,
                                                                                              y_test)


        #selected_features = self._selected_top_features.split(',')
        # selected_features = self._selected_top_features.split('||')
        selected_features = self._selected_top_features
        num_of_features = len(selected_features)

        X_original_train = X_original_train[selected_features]
        X_test = X_test[selected_features]

        performance_results = []
        for iteration in range(1, self._iterations + 1):
            begin_time = time.time()

            classifier, classifier_name = self._create_random_forest_classifier_by_selected_parameters(
                self._selected_estimators,
                self._selected_max_depth,
                self._selected_min_samples_split,
                self._selected_min_samples_leaf,
                self._selected_max_leaf_nodes)

            classifier.fit(X_original_train, y_original_train)

            performance_result = self._calculate_performance(classifier, X_original_train, X_test,
                                                             y_original_train, y_test)

            end_time = time.time()
            running_time = end_time - begin_time

            selected_parameters = (classifier_name, num_of_features, training_set_size, test_set_size,
                                   train_num_of_false_records, train_num_of_true_records,
                                   test_num_of_false_records, test_num_of_true_records,
                                   self._selected_estimators, self._selected_max_depth,
                                   self._selected_min_samples_split, self._selected_min_samples_leaf,
                                   self._selected_max_leaf_nodes, iteration, running_time, self._selected_top_features)
            performance_with_parameters = performance_result + selected_parameters

            performance_results.append(performance_with_parameters)

        performance_df = pd.DataFrame(performance_results, columns=['AUC Training', 'Accuracy Training',
                                                                    'F1 Training', 'Precision Training',
                                                                    'Recall Training', 'AUC Validation',
                                                                    'Accuracy Validation', 'F1 validation',
                                                                    'Precision Validation', 'Recall Validation',
                                                                    'Classifier', 'Num of Features', 'Training Set Size',
                                                                    'Test Set Size', '#False(Training)', '#True(Training)',
                                                                    '#False(Test)', '#True(Test)',
                                                                    'Estimators', 'Max Depth', 'Min Samples Split',
                                                                    'Min Samples Leaf', 'Max Leaf Nodes',
                                                                    'Iteration', 'Time', 'Top Features'])

        performance_df.to_csv(self._output_path + "RandomForest_Random_Selection_Validation_Results_on_Training_and_Test_sets.csv")

    def _calculate_performance(self, classifier, X_train, X_test, y_train, y_test):
        auc_training = roc_auc_score(y_train, classifier.predict_proba(X_train)[:, 1])
        auc_test = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])

        #fpr, tpr, thresholds = roc_curve(y_train, classifier.predict_proba(X_train)[:, 1])

        accuracy_train = accuracy_score(y_train, classifier.predict(X_train))
        accuracy_test = accuracy_score(y_test, classifier.predict(X_test))

        f1_train = f1_score(y_train, classifier.predict(X_train))
        f1_test = f1_score(y_test, classifier.predict(X_test))

        precision_train = precision_score(y_train, classifier.predict(X_train))
        precision_test = precision_score(y_test, classifier.predict(X_test))

        recall_train = recall_score(y_train, classifier.predict(X_train))
        recall_test = recall_score(y_test, classifier.predict(X_test))

        performance_result = (auc_training, accuracy_train, f1_train, precision_train, recall_train, auc_test,
                              accuracy_test, f1_test, precision_test, recall_test)

        return performance_result


    def _split_vals(self, a, n):
        return a[:n].copy(), a[n:].copy()

    def _remove_features_and_preprocess_dataset(self, df):
        df = self._remove_selected_features(df)
        df = self._preprocess_dataset(df)

        return df

    def _remove_selected_features(self, df):
        # Remove ids features
        for remove_feature in self._remove_features:
            df.pop(remove_feature)
        return df

    def _preprocess_dataset(self, df):
        # Convert the verdict date to date type
        df[self._date_field_name] = pd.to_datetime(
            df[self._date_field_name])

        df = df.fillna(0)

        # Convert labels to binary
        df.replace(to_replace=True, value='0', inplace=True)
        df.replace(to_replace=False, value='1', inplace=True)

        df[self._targeted_class_field_name] = df[self._targeted_class_field_name].apply(int)
        # y_train = y_train.apply(int)
        # y_test = y_test.apply(int)

        return df

    def _create_training_and_test_sets(self, df, y):
        X_original_train, X_test = self._split_vals(df, self._training_set_num_of_records)
        y_original_train, y_test = self._split_vals(y, self._training_set_num_of_records)

        return X_original_train, X_test, y_original_train, y_test

    def _create_training_and_test_sets_and_remove_features(self, df, y):
        X_original_train, X_test, y_original_train, y_test = self._create_training_and_test_sets(df, y)
        # Remove targeted class name
        X_original_train.pop(self._targeted_class_field_name)
        X_test.pop(self._targeted_class_field_name)

        # remove sorted by date field
        if self._sort_by_date == True:
            X_original_train.pop(self._date_field_name)
            X_test.pop(self._date_field_name)

        return X_original_train, X_test, y_original_train, y_test

    def _create_training_and_validation_sets(self, X_original_train, y_original_train):
        training_set_size = self._training_set_num_of_records - self._validation_set_num_of_records

        X_train, X_validation = self._split_vals(X_original_train, training_set_size)
        y_train, y_validation = self._split_vals(y_original_train, training_set_size)

        return X_train, X_validation, y_train, y_validation

    def _find_best_features(self, classifier, X_train, num_of_features):
        features_df = pd.DataFrame()
        features_df["score"] = classifier.feature_importances_
        features_df["name"] = X_train.columns

        important_features_df = features_df[features_df["score"] > 0]
        important_features_df = important_features_df.sort_values(by='score', ascending=False)

        top_features = important_features_df["name"].values[:num_of_features]

        return top_features

    def _get_feature_importance_greater_than_zero(self, classifier, X_train):
        features_df = pd.DataFrame()
        features_df["score"] = classifier.feature_importances_
        features_df["name"] = X_train.columns

        important_features_df = features_df[features_df["score"] > 0]
        important_features_df = important_features_df.sort_values(by='score', ascending=False)
        return important_features_df

    def _get_top_n_features(self, important_features_df, n):
        top_features = important_features_df["name"].values[:n]
        return top_features

    def _get_num_of_true_and_false_records(self, y):
        counts = y.value_counts()
        # 1 = False, 0 = True
        num_of_false_records = counts[1]
        num_of_true_records = counts[0]

        return num_of_false_records, num_of_true_records

    def _read_csv_file_and_remove_features_and_preprocess_dataset(self):
        claim_features_df = pd.read_csv(self._input_path + self._dataset_csv_file_name)

        if self._sort_by_date == True:
            # Sort by date
            claim_features_df = claim_features_df.sort_values(by=self._date_field_name)

        claim_features_df = self._remove_features_and_preprocess_dataset(claim_features_df)
        return claim_features_df


    def _read_csv_file_and_preprocess_dataset(self):
        claim_features_df = pd.read_csv(self._input_path + self._dataset_csv_file_name)

        if self._sort_by_date == True:
            # Sort by date
            claim_features_df = claim_features_df.sort_values(by=self._date_field_name)

        claim_features_df = self._preprocess_dataset(claim_features_df)
        return claim_features_df

    def _read_csv_file_and_divide_to_train_and_test_sets(self):
        claim_features_df = self._read_csv_file_and_remove_features_and_preprocess_dataset()

        claim_features_df.replace(to_replace=np.Infinity, value='-1', inplace=True)

        y = claim_features_df[self._targeted_class_field_name]
        X_original_train, X_test, y_original_train, y_test = self._create_training_and_test_sets_and_remove_features(
            claim_features_df, y)

        return X_original_train, X_test, y_original_train, y_test

    # Can be also train and validation not only train and test
    def _get_set_sizes_and_true_and_false_records_in_each_set(self, y_original_train, y_test):
        training_set_size = len(y_original_train)
        test_set_size = len(y_test)

        train_num_of_false_records, train_num_of_true_records = self._get_num_of_true_and_false_records(
            y_original_train)
        test_num_of_false_records, test_num_of_true_records = self._get_num_of_true_and_false_records(
            y_test)

        return training_set_size, test_set_size, train_num_of_false_records, train_num_of_true_records, \
               test_num_of_false_records, test_num_of_true_records

    def _create_default_random_forest_classifier_and_calculate_feature_importance(self, X_original_train, y_original_train):
        default_classifier = RandomForestClassifier()
        default_classifier.fit(X_original_train, y_original_train)
        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_Random_Forest.csv")

        return important_features_df

    def _create_default_xgboost_classifier_and_calculate_feature_importance(self, X_original_train, y_original_train):
        default_classifier = xgb.XGBClassifier()

        X_original_train = self._convert_df_columns_to_float(X_original_train)

        default_classifier.fit(X_original_train, y_original_train)
        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_XGBoost.csv")

        return important_features_df


    def _create_default_adaboost_classifier_and_calculate_feature_importance(self, X_original_train, y_original_train):
        default_classifier = AdaBoostClassifier()

        default_classifier.fit(X_original_train, y_original_train)
        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_AdaBoost.csv")

        return important_features_df

    def _add_random_forest_parameters(self):
        parameters = []
        parameters.append(self._random_forest_estimators)
        parameters.append(self._random_forest_max_depth)
        parameters.append(self._random_forest_min_samples_split)
        parameters.append(self._random_forest_min_samples_leaf)
        parameters.append(self._random_forest_max_leaf_nodes)
        parameters.append(self._num_of_features)

        return parameters

    def _add_xgboost_parameters(self):
        parameters = []
        parameters.append(self._xgboost_estimators)
        parameters.append(self._xgboost_max_depth)
        parameters.append(self._xgboost_learning_rate)
        parameters.append(self._num_of_features)

        return parameters

    def _add_adaboost_parameters(self):
        parameters = []
        parameters.append(self._adaboost_estimators)
        parameters.append(self._adaboost_learning_rate)
        parameters.append(self._num_of_features)

        return parameters

    def _create_combinations_by_parameters(self, parameters):
        combinations = list(itertools.product(*parameters))
        return combinations

    def _add_random_forest_parameters_and_create_combinations(self):
        parameters = self._add_random_forest_parameters()
        combinations = self._create_combinations_by_parameters(parameters)
        return combinations

    def _add_xgboost_parameters_and_create_combinations(self):
        parameters = self._add_xgboost_parameters()
        combinations = self._create_combinations_by_parameters(parameters)
        return combinations

    def _add_adaboost_parameters_and_create_combinations(self):
        parameters = self._add_adaboost_parameters()
        combinations = self._create_combinations_by_parameters(parameters)
        return combinations

    def _read_random_forest_combination(self, combination):
        random_forest_estimator = combination[0]
        random_forest_max_depth = combination[1]
        random_forest_min_samples_split = combination[2]
        random_forest_min_samples_leaf = combination[3]
        random_forest_max_leaf_nodes = combination[4]
        num_of_features = combination[5]

        return random_forest_estimator, random_forest_max_depth, random_forest_min_samples_split, \
               random_forest_min_samples_leaf, random_forest_max_leaf_nodes, num_of_features

    def _create_random_forest_classifier_by_selected_parameters(self, random_forest_estimator, random_forest_max_depth,
                                                                random_forest_min_samples_split,
                                                                random_forest_min_samples_leaf,
                                                                random_forest_max_leaf_nodes):
        classifier = RandomForestClassifier(n_estimators=random_forest_estimator,
                                            max_depth=random_forest_max_depth,
                                            min_samples_split=random_forest_min_samples_split,
                                            min_samples_leaf=random_forest_min_samples_leaf,
                                            max_leaf_nodes=random_forest_max_leaf_nodes)
        classifier_name = classifier.__class__.__name__

        return classifier, classifier_name


    def _create_xgboost_classifier_by_selected_parameters(self, xgboost_estimator, xgboost_max_depth,
                                                          xgboost_learning_rate,):
        classifier = xgb.XGBClassifier(n_estimators=xgboost_estimator,
                                       max_depth=xgboost_max_depth,
                                       learning_rate=xgboost_learning_rate)
        classifier_name = classifier.__class__.__name__

        return classifier, classifier_name

    def _create_adaboost_classifier_by_selected_parameters(self, adaboost_estimator, adaboost_learning_rate,):
        classifier = AdaBoostClassifier(n_estimators=adaboost_estimator,
                                       learning_rate=adaboost_learning_rate)
        classifier_name = classifier.__class__.__name__

        return classifier, classifier_name


    # Also can be train-test or train-validation
    def _set_sets_by_top_features(self, important_features_df, num_of_features, X_train, X_validation):
        top_features = self._get_top_n_features(important_features_df, num_of_features)
        X_train_ = X_train[top_features]
        X_validation_ = X_validation[top_features]

        top_feature_names = '||'.join(top_features)

        return X_train_, X_validation_, top_feature_names

    def _train_classifier_and_calculate_performance_on_two_given_sets(self, classifier, X_train_, X_validation_,
                                                                      y_train, y_validation):
        classifier.fit(X_train_, y_train)

        performance_result = self._calculate_performance(classifier, X_train_, X_validation_,
                                                         y_train, y_validation)

        return performance_result

    def _add_random_forest_parameters_to_performance_result(self, performance_result, classifier_name, num_of_features,
                                                            training_set_size, validation_set_size, train_num_of_false_records,
                                                            train_num_of_true_records, validation_num_of_false_records,
                                                            validation_num_of_true_records, random_forest_estimator,
                                                            random_forest_max_depth, random_forest_min_samples_split,
                                                            random_forest_min_samples_leaf, random_forest_max_leaf_nodes, iteration,
                                                            running_time, top_feature_names):

        selected_parameters = (classifier_name, num_of_features,
                               training_set_size, validation_set_size,
                               train_num_of_false_records, train_num_of_true_records,
                               validation_num_of_false_records, validation_num_of_true_records,
                               random_forest_estimator, random_forest_max_depth,
                               random_forest_min_samples_split, random_forest_min_samples_leaf,
                               random_forest_max_leaf_nodes, iteration, running_time, top_feature_names)
        performance_with_parameters = performance_result + selected_parameters

        return performance_with_parameters

    def _read_combination_create_random_forest_classifier_train_and_calculate_performance(self, combination,
                                                                                          important_features_df,
                                                                                          X_train, X_validation,
                                                                                          y_train, y_validation,
                                                                                          training_set_size,
                                                                                          validation_set_size,
                                                                                          train_num_of_false_records,
                                                                                          train_num_of_true_records,
                                                                                          validation_num_of_false_records,
                                                                                          validation_num_of_true_records,
                                                                                          iteration):
        random_forest_estimator, random_forest_max_depth, random_forest_min_samples_split, \
        random_forest_min_samples_leaf, random_forest_max_leaf_nodes, num_of_features = self._read_random_forest_combination(
            combination)

        begin_time = time.time()
        classifier, classifier_name = self._create_random_forest_classifier_by_selected_parameters(
            random_forest_estimator,
            random_forest_max_depth,
            random_forest_min_samples_split,
            random_forest_min_samples_leaf,
            random_forest_max_leaf_nodes)

        X_train_, X_validation_, top_feature_names = self._set_sets_by_top_features(important_features_df,
                                                                                    num_of_features,
                                                                                    X_train, X_validation)

        performance_result = self._train_classifier_and_calculate_performance_on_two_given_sets(classifier,
                                                                                                X_train_,
                                                                                                X_validation_,
                                                                                                y_train,
                                                                                                y_validation)

        end_time = time.time()
        running_time = end_time - begin_time

        performance_with_parameters = self._add_random_forest_parameters_to_performance_result(performance_result,
                                                                                               classifier_name,
                                                                                               num_of_features,
                                                                                               training_set_size,
                                                                                               validation_set_size,
                                                                                               train_num_of_false_records,
                                                                                               train_num_of_true_records,
                                                                                               validation_num_of_false_records,
                                                                                               validation_num_of_true_records,
                                                                                               random_forest_estimator,
                                                                                               random_forest_max_depth,
                                                                                               random_forest_min_samples_split,
                                                                                               random_forest_min_samples_leaf,
                                                                                               random_forest_max_leaf_nodes,
                                                                                               iteration,
                                                                                               running_time,
                                                                                               top_feature_names)

        return performance_with_parameters


    def _convert_df_columns_to_float(self, df):
        column_names = df.columns.values
        for column_name in column_names:
            feature_series = df[column_name]
            feature_series = feature_series.astype(np.float64)
            # feature_series = feature_series.astype(np.int64)
            df[column_name] = feature_series
        return df

    def _read_combination_create_xgboost_classifier_train_and_calculate_performance(self, combination,
                                                                                    important_features_df,
                                                                                    X_original_train, X_test,
                                                                                    y_original_train, y_test,
                                                                                    train_set_size,
                                                                                    test_set_size,
                                                                                    train_num_of_false_records,
                                                                                    train_num_of_true_records,
                                                                                    test_num_of_false_records,
                                                                                    test_num_of_true_records,
                                                                                    iteration):
        xgboost_estimator, xgboost_max_depth, xgboost_learning_rate, num_of_features = self._read_xgboost_combination(
            combination)

        begin_time = time.time()
        classifier, classifier_name = self._create_xgboost_classifier_by_selected_parameters(
            xgboost_estimator, xgboost_max_depth, xgboost_learning_rate)

        X_original_train, X_test, top_feature_names = self._set_sets_by_top_features(important_features_df,
                                                                                    num_of_features,
                                                                                    X_original_train, X_test)

        performance_result = self._train_classifier_and_calculate_performance_on_two_given_sets(classifier,
                                                                                                X_original_train,
                                                                                                X_test,
                                                                                                y_original_train,
                                                                                                y_test)

        end_time = time.time()
        running_time = end_time - begin_time

        performance_with_parameters = self._add_xgboost_parameters_to_performance_result(performance_result,
                                                                                               classifier_name,
                                                                                               num_of_features,
                                                                                               train_set_size,
                                                                                               test_set_size,
                                                                                               train_num_of_false_records,
                                                                                               train_num_of_true_records,
                                                                                               test_num_of_false_records,
                                                                                               test_num_of_true_records,
                                                                                               xgboost_estimator,
                                                                                               xgboost_max_depth,
                                                                                               xgboost_learning_rate,
                                                                                               iteration,
                                                                                               running_time,
                                                                                               top_feature_names)

        return performance_with_parameters


    def _read_combination_create_adaboost_classifier_train_and_calculate_performance(self, combination,
                                                                                    important_features_df,
                                                                                    X_original_train, X_test,
                                                                                    y_original_train, y_test,
                                                                                    train_set_size,
                                                                                    test_set_size,
                                                                                    train_num_of_false_records,
                                                                                    train_num_of_true_records,
                                                                                    test_num_of_false_records,
                                                                                    test_num_of_true_records,
                                                                                    iteration):
        adaboost_estimator, adaboost_learning_rate, num_of_features = self._read_adaboost_combination(
            combination)

        begin_time = time.time()
        classifier, classifier_name = self._create_adaboost_classifier_by_selected_parameters(
            adaboost_estimator, adaboost_learning_rate)

        X_original_train, X_test, top_feature_names = self._set_sets_by_top_features(important_features_df,
                                                                                    num_of_features,
                                                                                    X_original_train, X_test)

        performance_result = self._train_classifier_and_calculate_performance_on_two_given_sets(classifier,
                                                                                                X_original_train,
                                                                                                X_test,
                                                                                                y_original_train,
                                                                                                y_test)

        end_time = time.time()
        running_time = end_time - begin_time

        performance_with_parameters = self._add_adaboost_parameters_to_performance_result(performance_result,
                                                                                               classifier_name,
                                                                                               num_of_features,
                                                                                               train_set_size,
                                                                                               test_set_size,
                                                                                               train_num_of_false_records,
                                                                                               train_num_of_true_records,
                                                                                               test_num_of_false_records,
                                                                                               test_num_of_true_records,
                                                                                               adaboost_estimator,
                                                                                               adaboost_learning_rate,
                                                                                               iteration,
                                                                                               running_time,
                                                                                               top_feature_names)

        return performance_with_parameters



    def _read_xgboost_combination(self, combination):

        # max_depth = 3, learning_rate = 0.1,
        # n_estimators = 100, silent = True,
        # objective = "binary:logistic",
        # nthread = -1, gamma = 0, min_child_weight = 1,
        # max_delta_step = 0, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1,
        # reg_alpha = 0, reg_lambda = 1, scale_pos_weight = 1,
        # base_score = 0.5
        #

        xgboost_estimator = combination[0]
        xgboost_max_depth = combination[1]
        xgboost_learning_rate = combination[2]
        # xgboost_gamma = combination[3]
        # xgboost_min_child_weight = combination[4]
        # xgboost_max_delta_step = combination[5]
        # xgboost_reg_alpha = combination[6]
        # xgboost_reg_lambda = combination[7]
        # xgboost_base_score = combination[8]
        num_of_features = combination[3]

        return xgboost_estimator, xgboost_max_depth, xgboost_learning_rate, num_of_features

    def _read_adaboost_combination(self, combination):

        adaboost_estimator = combination[0]
        adaboost_learning_rate = combination[1]
        num_of_features = combination[2]

        return adaboost_estimator, adaboost_learning_rate, num_of_features

    def _add_xgboost_parameters_to_performance_result(self, performance_result, classifier_name, num_of_features,
                                                      train_set_size, test_set_size, train_num_of_false_records,
                                                      train_num_of_true_records, test_num_of_false_records,
                                                      test_num_of_true_records, xgboost_estimator, xgboost_max_depth,
                                                      xgboost_learning_rate, iteration, running_time,
                                                      top_feature_names):
        selected_parameters = (classifier_name, num_of_features,
                               train_set_size, test_set_size,
                               train_num_of_false_records, train_num_of_true_records,
                               test_num_of_false_records, test_num_of_true_records,
                               xgboost_estimator, xgboost_max_depth,
                               xgboost_learning_rate, iteration, running_time, top_feature_names)
        performance_with_parameters = performance_result + selected_parameters

        return performance_with_parameters

    def _add_adaboost_parameters_to_performance_result(self, performance_result, classifier_name, num_of_features,
                                                      train_set_size, test_set_size, train_num_of_false_records,
                                                      train_num_of_true_records, test_num_of_false_records,
                                                      test_num_of_true_records, adaboost_estimator,
                                                      adaboost_learning_rate, iteration, running_time,
                                                      top_feature_names):
        selected_parameters = (classifier_name, num_of_features,
                               train_set_size, test_set_size,
                               train_num_of_false_records, train_num_of_true_records,
                               test_num_of_false_records, test_num_of_true_records,
                               adaboost_estimator, adaboost_learning_rate, iteration, running_time, top_feature_names)
        performance_with_parameters = performance_result + selected_parameters

        return performance_with_parameters

    def _create_default_classifier_and_calculate_feature_importance(self, X_original_train, y_original_train):
        default_classifier = globals()[self._default_classifier]()
        classifier_name = default_classifier.__class__.__name__

        X_original_train = self._convert_df_columns_to_float(X_original_train)

        default_classifier.fit(X_original_train, y_original_train)
        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_{0}.csv".format(classifier_name))

        return important_features_df

    def _create_default_adaboost_classifier_and_calculate_feature_importance(self, X_original_train, y_original_train):
        default_classifier = AdaBoostClassifier()
        classifier_name = default_classifier.__class__.__name__

        X_original_train = self._convert_df_columns_to_float(X_original_train)

        default_classifier.fit(X_original_train, y_original_train)
        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(self._output_path + "Training_Feature_Importance_No_Validation_{0}.csv".format(classifier_name))

        return important_features_df

    def _create_default_decision_tree_classifier_and_calculate_feature_importance(self, X_original_train,
                                                                                  y_original_train):
        default_classifier = tree.DecisionTreeClassifier()
        classifier_name = default_classifier.__class__.__name__

        X_original_train = self._convert_df_columns_to_float(X_original_train)

        default_classifier.fit(X_original_train, y_original_train)
        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(
            self._output_path + "Training_Feature_Importance_No_Validation_{0}.csv".format(classifier_name))

        return important_features_df

    def _add_decision_tree_parameters_and_create_combinations(self):
        parameters = self._add_decision_tree_parameters()
        combinations = self._create_combinations_by_parameters(parameters)
        return combinations

    def _add_decision_tree_parameters(self):
        parameters = []
        parameters.append(self._decision_tree_max_depth)
        parameters.append(self._decision_tree_min_samples_split)
        parameters.append(self._decision_tree_min_samples_leaf)
        parameters.append(self._decision_tree_max_leaf_nodes)
        parameters.append(self._num_of_features)

        return parameters

    def _read_combination_create_decision_tree_classifier_train_and_calculate_performance(self, combination,
                                                                                important_features_df, X_original_train,
                                                                                X_test, y_original_train, y_test,
                                                                                training_set_size, test_set_size,
                                                                                train_num_of_false_records,
                                                                                train_num_of_true_records,
                                                                                test_num_of_false_records,
                                                                                test_num_of_true_records, iteration):
        decision_tree_max_depth, decision_tree_min_samples_split, decision_tree_min_samples_leaf, \
        decision_tree_max_leaf_nodes, num_of_features = self._read_decision_tree_combination(
            combination)

        begin_time = time.time()
        classifier, classifier_name = self._create_decision_tree_classifier_by_selected_parameters(
            decision_tree_max_depth, decision_tree_min_samples_split, decision_tree_min_samples_leaf,
            decision_tree_max_leaf_nodes)

        X_original_train, X_test, top_feature_names = self._set_sets_by_top_features(important_features_df,
                                                                                     num_of_features,
                                                                                     X_original_train, X_test)

        performance_result = self._train_classifier_and_calculate_performance_on_two_given_sets(classifier,
                                                                                                X_original_train,
                                                                                                X_test,
                                                                                                y_original_train,
                                                                                                y_test)

        end_time = time.time()
        running_time = end_time - begin_time

        performance_with_parameters = self._add_decision_tree_parameters_to_performance_result(performance_result,
                                                                                          classifier_name,
                                                                                          num_of_features,
                                                                                          training_set_size,
                                                                                          test_set_size,
                                                                                          train_num_of_false_records,
                                                                                          train_num_of_true_records,
                                                                                          test_num_of_false_records,
                                                                                          test_num_of_true_records,
                                                                                          decision_tree_max_depth,
                                                                                          decision_tree_min_samples_split,
                                                                                          decision_tree_min_samples_leaf,
                                                                                          decision_tree_max_leaf_nodes,
                                                                                          iteration,
                                                                                          running_time,
                                                                                          top_feature_names)

        return performance_with_parameters

    def _read_decision_tree_combination(self, combination):
        parameters = []
        parameters.append(self._decision_tree_max_depth)
        parameters.append(self._decision_tree_min_samples_split)
        parameters.append(self._decision_tree_min_samples_leaf)
        parameters.append(self._decision_tree_max_leaf_nodes)
        parameters.append(self._num_of_features)

        decision_tree_max_depth = combination[0]
        decision_tree_min_samples_split = combination[1]
        decision_tree_min_samples_leaf = combination[2]
        decision_tree_max_leaf_nodes = combination[3]
        num_of_features = combination[4]

        return decision_tree_max_depth, decision_tree_min_samples_split, decision_tree_min_samples_leaf, \
               decision_tree_max_leaf_nodes, num_of_features

    def _create_decision_tree_classifier_by_selected_parameters(self, decision_tree_max_depth,
                                                                decision_tree_min_samples_split,
                                                                decision_tree_min_samples_leaf,
                                                                decision_tree_max_leaf_nodes):

        classifier = tree.DecisionTreeClassifier(max_depth=decision_tree_max_depth,
                                                 min_samples_split=decision_tree_min_samples_split,
                                                 min_samples_leaf=decision_tree_min_samples_leaf,
                                                 max_leaf_nodes=decision_tree_max_leaf_nodes)
        classifier_name = classifier.__class__.__name__

        return classifier, classifier_name

    def _add_decision_tree_parameters_to_performance_result(self, performance_result, classifier_name, num_of_features,
                                                            training_set_size, test_set_size,
                                                            train_num_of_false_records, train_num_of_true_records,
                                                            test_num_of_false_records, test_num_of_true_records,
                                                            decision_tree_max_depth, decision_tree_min_samples_split,
                                                            decision_tree_min_samples_leaf,
                                                            decision_tree_max_leaf_nodes, iteration, running_time,
                                                            top_feature_names):
        selected_parameters = (classifier_name, num_of_features,
                               training_set_size, test_set_size,
                               train_num_of_false_records, train_num_of_true_records,
                               test_num_of_false_records, test_num_of_true_records,
                               decision_tree_max_depth, decision_tree_min_samples_split,
                               decision_tree_min_samples_leaf,
                               decision_tree_max_leaf_nodes, iteration, running_time, top_feature_names)
        performance_with_parameters = performance_result + selected_parameters

        return performance_with_parameters

    def _create_default_svm_classifier_and_calculate_feature_importance(self, X_original_train, y_original_train):
        default_classifier = svm.SVC(kernel = 'linear')
        classifier_name = default_classifier.__class__.__name__

        default_classifier.fit(X_original_train, y_original_train)

        x = pd.Series(abs(default_classifier.coef_[0]), index=X_original_train.columns).nlargest(10)


        important_features_df = self._get_feature_importance_greater_than_zero(default_classifier, X_original_train)
        important_features_df.to_csv(
            self._output_path + "Training_Feature_Importance_No_Validation_{0}.csv".format(classifier_name))

        return important_features_df

    def _add_svm_parameters_and_create_combinations(self):
        parameters = self._add_svm_parameters()
        combinations = self._create_combinations_by_parameters(parameters)
        return combinations

    def _add_svm_parameters(self):
        parameters = []
        parameters.append(self._svm_c)
        parameters.append(self._svm_degree)
        parameters.append(self._num_of_features)

        return parameters

    def _read_combination_create_svm_classifier_train_and_calculate_performance(self, combination,
                                                                                important_features_df, X_original_train,
                                                                                X_test, y_original_train, y_test,
                                                                                training_set_size, test_set_size,
                                                                                train_num_of_false_records,
                                                                                train_num_of_true_records,
                                                                                test_num_of_false_records,
                                                                                test_num_of_true_records, iteration):

        svm_c, svm_degree, num_of_features = self._read_svm_combination(combination)

        begin_time = time.time()
        classifier, classifier_name = self._create_svm_classifier_by_selected_parameters(svm_c, svm_degree)

        X_original_train, X_test, top_feature_names = self._set_sets_by_top_features(important_features_df,
                                                                                     num_of_features,
                                                                                     X_original_train, X_test)

        performance_result = self._train_classifier_and_calculate_performance_on_two_given_sets(classifier,
                                                                                                X_original_train,
                                                                                                X_test,
                                                                                                y_original_train,
                                                                                                y_test)

        end_time = time.time()
        running_time = end_time - begin_time

        performance_with_parameters = self._add_svm_parameters_to_performance_result(performance_result,
                                                                                               classifier_name,
                                                                                               num_of_features,
                                                                                               training_set_size,
                                                                                               test_set_size,
                                                                                               train_num_of_false_records,
                                                                                               train_num_of_true_records,
                                                                                               test_num_of_false_records,
                                                                                               test_num_of_true_records,
                                                                                               svm_c, svm_degree,
                                                                                               iteration,
                                                                                               running_time,
                                                                                               top_feature_names)

        return performance_with_parameters

    def _read_svm_combination(self, combination):
        parameters = []
        parameters.append(self._svm_c)
        parameters.append(self._svm_degree)
        parameters.append(self._num_of_features)

        svm_c = combination[0]
        svm_degree = combination[1]
        num_of_features = combination[2]

        return svm_c, svm_degree, num_of_features

    def _create_svm_classifier_by_selected_parameters(self, svm_c, svm_degree):
        classifier = svm.SVC(C=svm_c, degree=svm_degree)
        classifier_name = classifier.__class__.__name__

        return classifier, classifier_name

    def _add_svm_parameters_to_performance_result(self, performance_result, classifier_name, num_of_features,
                                                  training_set_size, test_set_size, train_num_of_false_records,
                                                  train_num_of_true_records, test_num_of_false_records,
                                                  test_num_of_true_records, svm_c, svm_degree, iteration, running_time,
                                                  top_feature_names):
        selected_parameters = (classifier_name, num_of_features,
                               training_set_size, test_set_size,
                               train_num_of_false_records, train_num_of_true_records,
                               test_num_of_false_records, test_num_of_true_records,
                               svm_c, svm_degree, iteration, running_time, top_feature_names)
        performance_with_parameters = performance_result + selected_parameters

        return performance_with_parameters


