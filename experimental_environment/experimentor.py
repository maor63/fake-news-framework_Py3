import pickle
import timeit
from itertools import chain

import xgboost as xgb
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import tree
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from commons.commons import retreive_labeled_authors_dataframe, replace_nominal_class_to_numeric, \
    calculate_correctly_and_not_correctly_instances
from commons.consts import Classifiers
from commons.data_frame_creator import DataFrameCreator
from commons.method_executor import Method_Executor
import time
import pandas as pd
import os
from sklearn.externals import joblib
import numpy as np
# from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# from dataset_builder.feature_extractor.AdaRank.adarank import AdaRank
# from dataset_builder.feature_extractor.AdaRank.metrics import NDCGScorer


class Experimentor(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._targeted_class_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_name")
        self._classifier_type_names = self._config_parser.eval(self.__class__.__name__, "classifier_type_names")
        self._num_of_features_to_train = self._config_parser.eval(self.__class__.__name__, "num_of_features_to_train")
        self._k_for_fold = self._config_parser.eval(self.__class__.__name__, "k_for_fold")
        self._replace_missing_values = self._config_parser.eval(self.__class__.__name__, "replace_missing_values")
        self._optional_classes = self._config_parser.eval(self.__class__.__name__, "optional_classes")
        self._is_create_equal_number_of_records_per_class = self._config_parser.eval(self.__class__.__name__,
                                                                                     "is_create_equal_number_of_records_per_class")
        self._targeted_class_field_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_name")
        self._targeted_classes_dict = self._config_parser.eval(self.__class__.__name__, "targeted_classes_dict")
        self._index_field = self._config_parser.eval(self.__class__.__name__, "index_field")
        self._target_is_string_classes = self._config_parser.eval(self.__class__.__name__, "target_is_string_classes")
        self._removed_features = self._config_parser.eval(self.__class__.__name__, "removed_features")
        self._selected_features = self._config_parser.eval(self.__class__.__name__, "selected_features")
        self._full_path_model_directory = self._config_parser.eval(self.__class__.__name__, "full_path_model_directory")
        self._export_file_with_predictions_during_training = self._config_parser.eval(self.__class__.__name__,
                                                                                      "export_file_with_predictions_during_training")
        self._prepared_classifier_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                       "prepared_classifier_file_name")
        self._prepared_classifier_selected_features_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                                         "prepared_classifier_selected_features_file_name")
        self._targeted_classifier_name = self._config_parser.eval(self.__class__.__name__, "targeted_classifier_name")
        self._targeted_classifier_num_of_features = self._config_parser.eval(self.__class__.__name__,
                                                                             "targeted_classifier_num_of_features")

        # divide_records_by_trageted_classes
        self._divide_records_by_trageted_classes = self._config_parser.eval(self.__class__.__name__,
                                                                            "divide_records_by_trageted_classes")
        self._iterations = self._config_parser.eval(self.__class__.__name__, "iterations")
        self._divide_records_by_targeted_classes_experiments = self._config_parser.eval(self.__class__.__name__,
                                                                                        "divide_records_by_trageted_classes_experiments")

        self._training_percent = self._config_parser.eval(self.__class__.__name__, "training_percent")

    def setUp(self):
        if not os.path.isdir(self._full_path_model_directory):
            os.makedirs(self._full_path_model_directory)

    def extract_dataset(self):
        features_df = self._get_author_features_dataframe()
        features_df.to_csv(self._full_path_model_directory + "Fake_News_WoTC_Dataset.csv")
        print("Done!!!!")

    def predict_flickr_graphs_labels(self):
        author_feature_df = self._get_author_features_dataframe()
        selected_features_df = author_feature_df

        if self._target_is_string_classes:
            graphs_labels = author_feature_df[self._targeted_class_name].map(lambda x: unique_labels(x.split(',')))
            all_labels = list(chain(*graphs_labels))
            distinct_labels = unique_labels(all_labels)
            print(("distinct label count: {}".format(len(distinct_labels))))
            print("build label encoder")
            label_encoder = self.build_label_encoder(distinct_labels)
            print("convert label to one hot vector")
            label_encoding_map = self._build_label_to_encoding_map(distinct_labels, label_encoder)
            print('get photos ground truth encodings')
            encodings = self._labels_to_encodings(graphs_labels, label_encoder, label_encoding_map)
        else:
            graphs_labels = author_feature_df[self._targeted_class_name].map(lambda x: np.array(list(map(int, x.split(',')))))
            print(("distinct label count: {}".format(len(graphs_labels[0]))))
            encodings = np.array([np.array(lv) for lv in graphs_labels])

        selected_features_df.pop(self._targeted_class_name)
        X = selected_features_df
        Y = encodings
        print('start K fold cross validation')
        kf = KFold(n_splits=self._k_for_fold)
        precisions = []
        recalls = []
        mean_average_precisions = []
        f1_micros = []
        f1_macros = []
        i = 1
        for train, test in kf.split(X):
            start = timeit.default_timer()
            print(("iteration {}/{}".format(i, self._k_for_fold)))
            i += 1
            X_train, X_test = X.iloc[train], X.iloc[test]
            Y_train, Y_test = Y[train], Y[test]
            # classif = OneVsRestClassifier(SVC(verbose=True, max_iter=1000), n_jobs=1)
            # classif.fit(X_train, Y_train)
            # y_score = classif.decision_function(X_test)
            # y_pred = classif.predict(X_test)
            classif = RandomForestClassifier(10, max_depth=20, verbose=3, n_jobs=-1)
            classif.fit(X_train, Y_train)
            y_pred = classif.predict(X_test)

            # scorer = NDCGScorer(k=10)
            # model = AdaRank(max_iter=100, estop=10, scorer=scorer)
            # model.fit(X_train, Y_train, train)
            # pred = model.predict(X_test, test)
            # print 'NDCG@10: {}'.format(scorer(Y_test, y_pred, np.zeros(len(Y_test))).mean())

            precision, recall, _ = precision_recall_curve(Y_test.ravel(), y_pred.ravel())
            mean_average_precision = average_precision_score(Y_test, y_pred, average="micro")
            print(('MAP: {}'.format(mean_average_precision)))
            f1_micro = f1_score(Y_test, y_pred, average="micro")
            f1_macro = f1_score(Y_test, y_pred, average="macro")
            self._plot_precition_recall_curve(mean_average_precision, precision, recall)
            accuracy = accuracy_score(Y_test, y_pred)
            print(('ACC: {}'.format(np.mean(accuracy))))
            print(('f1_micro: {}'.format(f1_micro)))
            print(('f1_macro: {}'.format(f1_macro)))
            print(('LRAP: {}'.format(label_ranking_average_precision_score(Y_test, y_pred))))
            print(('LRL: {}'.format(label_ranking_loss(Y_test, y_pred))))
            print(('coverage_error: {}'.format(coverage_error(Y_test, y_pred))))

            mean_average_precisions.append(mean_average_precision)
            precisions.append(precision)
            recalls.append(recall)
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)
            end = timeit.default_timer()
            print(('!!!!!! run time {} sec !!!!!!!!!'.format(str(end - start))))

        precision = np.array([np.mean(values) for values in zip(*precisions)])
        recall = np.array([np.mean(values) for values in zip(*recalls)])

        mean_average_precision = np.mean(mean_average_precisions)
        print(('MAP score, micro-averaged over all classes: {0:0.2f}'
              .format(mean_average_precision)))
        f1_micro = np.mean(f1_micros)
        f1_macro = np.mean(f1_macros)
        print(("F1 micro: {}, F1 macro: {}".format(f1_micro, f1_macro)))
        self._plot_precition_recall_curve(mean_average_precision, precision, recall)
        pass

    def _plot_precition_recall_curve(self, average_precision, precision, recall):
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        title_text = 'MAP score, micro-averaged over all classes: MAP={0:0.2f}'.format(
            average_precision)
        plt.title(title_text)
        plt.show()

    def _calc_precision_recall_curve(self, Y_test, n_classes, y_score):
        precision = dict()
        recall = dict()
        # for i in range(n_classes):
        #     precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        # A "micro-average": quantifying score on all classes jointly
        precision, recall, _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
        return precision, recall

    def _labels_to_encodings(self, graphs_labels, label_encoder, label_encoding_map):
        encodings = []
        for labels in graphs_labels:
            indecis = label_encoder.transform(labels)
            encoding = np.add.reduce(label_encoding_map[indecis], 0)
            encodings.append(encoding)
        encodings = np.array(encodings)
        return encodings

    def _build_label_to_encoding_map(self, distinct_labels, label_encoder):
        integer_encoded = label_encoder.transform(distinct_labels)
        label_encoding_map = np.zeros((len(integer_encoded), len(integer_encoded)))
        label_encoding_map[np.arange(len(integer_encoded)), integer_encoded] = 1
        return label_encoding_map

    def build_label_encoder(self, distinct_labels):
        label_encoder = LabelEncoder()
        label_encoder.fit(distinct_labels)
        return label_encoder

    def check_model_is_overfitting_cross_validation(self):
        author_feature_df = self._get_author_features_dataframe()
        labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                 author_feature_df)

        result_performance_tuples = []

        labeled_features_df, targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
            labeled_features_df)

        labeled_features_df = labeled_features_df.convert_objects(convert_numeric=True)
        labeled_features_df = labeled_features_df.fillna(0)

        training_auc = []
        training_accuracy = []
        training_f1 = []
        training_precision = []
        training_recall = []

        test_auc = []
        test_accuracy = []
        test_f1 = []
        test_precision = []
        test_recall = []

        for num_of_features in self._num_of_features_to_train:
            targeted_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                labeled_features_df, targeted_class_series, num_of_features)

            dataframe_column_names = '||'.join(dataframe_column_names)

            for classifier_type_name in self._classifier_type_names:

                selected_classifier = self._select_classifier_by_type(classifier_type_name)

                if selected_classifier is not None:

                    if classifier_type_name == "XGBoost":
                        targeted_dataframe = self._set_dataframe_columns_types(targeted_dataframe)

                    kf = KFold(n_splits=self._k_for_fold)
                    for train_indexes, test_indexes in kf.split(targeted_dataframe):
                        training_set_df, test_set_df, training_class, test_class = self._create_train_and_test_dataframes_and_classes(
                            targeted_dataframe, train_indexes, test_indexes, targeted_class_series)

                        selected_classifier.fit(training_set_df, training_class)

                        test_predictions = selected_classifier.predict(test_set_df)
                        test_predictions_proba = selected_classifier.predict_proba(test_set_df)

                        auc_score, accuracy, f1, precision, recall = self._calculate_performance_and_return_results(
                            test_class, test_predictions, test_predictions_proba)

                        test_auc.append(auc_score)
                        test_accuracy.append(accuracy)
                        test_f1.append(f1)
                        test_precision.append(precision)
                        test_recall.append(recall)

                        training_predictions = selected_classifier.predict(training_set_df)
                        training_predictions_prob = selected_classifier.predict_proba(training_set_df)

                        auc_score, accuracy, f1, precision, recall = self._calculate_performance_and_return_results(
                            training_class, training_predictions, training_predictions_prob)

                        training_auc.append(auc_score)
                        training_accuracy.append(accuracy)
                        training_f1.append(f1)
                        training_precision.append(precision)
                        training_recall.append(recall)

        average_auc, average_accuracy, average_f1, average_precision, average_recall = self._calculate_average_performance(
            test_auc, test_accuracy, test_f1, test_precision, test_recall)
        result_tuple = (
            "Random Forest", 163, "test", average_auc, average_accuracy, average_f1, average_precision, average_recall)
        result_performance_tuples.append(result_tuple)

        average_auc, average_accuracy, average_f1, average_precision, average_recall = self._calculate_average_performance(
            training_auc, training_accuracy, training_f1, training_precision, training_recall)
        result_tuple = ("Random Forest", 163, "training", average_auc, average_accuracy, average_f1, average_precision,
                        average_recall)
        result_performance_tuples.append(result_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Classifier', 'Num_of_Features', 'set_type', 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall'])

        performance_result_df.to_csv(
            self._full_path_model_directory + "Classifier_Performance_check_whether_model_is_overfitting.csv")

        print("Done!!!!")

    def performance_function_training_size_order_by_num_of_posts_no_cross_validation(self):
        author_feature_df = self._get_author_features_dataframe()
        original_labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                          author_feature_df)

        claim_tuples = self._db.get_claims_tuples()
        claims_df = pd.DataFrame(claim_tuples, columns=["claim_id", "title", "description", "url", "verdict_date",
                                                        "keywords", "domain", "verdict", "category", "claim_topic",
                                                        "claim_post_id", "claim_ext_id", "sub_category",
                                                        "main_category",
                                                        "secondary_category"])

        claim_ordered_by_num_of_posts_tuples = self._db.get_claim_ordered_by_num_of_posts()
        claim_ordered_by_num_of_posts_df = pd.DataFrame(claim_ordered_by_num_of_posts_tuples,
                                                        columns=["claim_id", "num_of_posts"])

        claims_df = self._replace_nominal_values_to_numeric(claims_df)

        real_claims_df = claims_df[claims_df['verdict'] == 0]
        fake_claims_df = claims_df[claims_df['verdict'] == 1]

        real_claim_ordered_by_num_of_posts_df = pd.merge(claim_ordered_by_num_of_posts_df, real_claims_df,
                                                         on=['claim_id'], how='inner')

        fake_claim_ordered_by_num_of_posts_df = pd.merge(claim_ordered_by_num_of_posts_df, fake_claims_df,
                                                         on=['claim_id'], how='inner')

        result_performance_tuples = []
        for class_num_of_records_dict in self._divide_records_by_targeted_classes_experiments:
            for iteration in range(1, self._iterations + 1):

                # for targeted_class_, num_of_records in class_num_of_records_dict.iteritems():
                #     targeted_class_num_recods_tuple = (targeted_class_, self._targeted_classes_dict[targeted_class_], num_of_records)
                #     targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuples = []
                keys = list(class_num_of_records_dict.keys())
                values = list(class_num_of_records_dict.values())
                print(("#records = {0}-{1}, iteration = {2}".format(values[0], values[1], iteration)))
                targeted_class_num_recods_tuple = (keys[0], self._targeted_classes_dict[keys[0]], values[0])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuple = (keys[1], self._targeted_classes_dict[keys[1]], values[1])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                training_set_df, test_set_df = self._create_training_and_test_df_by_real_and_fake_claims_ordered_by_posts_dict(
                    class_num_of_records_dict, original_labeled_features_df,
                    real_claim_ordered_by_num_of_posts_df, fake_claim_ordered_by_num_of_posts_df)

                training_set_df, training_set_targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
                    training_set_df)
                training_set_df = training_set_df.convert_objects(convert_numeric=True)
                training_set_df = training_set_df.fillna(0)

                test_set_df, test_set_targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
                    test_set_df)
                test_set_df = test_set_df.convert_objects(convert_numeric=True)
                test_set_df = test_set_df.fillna(0)

                for num_of_features in self._num_of_features_to_train:
                    targeted_training_set_df, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                        training_set_df, training_set_targeted_class_series, num_of_features)

                    targeted_test_set_df = pd.DataFrame(test_set_df, columns=dataframe_column_names)
                    dataframe_column_names = '||'.join(dataframe_column_names)
                    for classifier_type_name in self._classifier_type_names:
                        begin_time = time.time()

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)

                        if selected_classifier is not None:

                            if classifier_type_name == "XGBoost":
                                targeted_training_set_df = self._set_dataframe_columns_types(targeted_training_set_df)

                            selected_classifier.fit(targeted_training_set_df, training_set_targeted_class_series)

                            predictions = selected_classifier.predict(targeted_test_set_df)
                            predictions_prob = selected_classifier.predict_proba(targeted_test_set_df)

                            # if self._export_file_with_predictions_during_training:
                            #     self._export_confidence_performance_with_tweets(index_field_series, classifier_type_name,
                            #                                             num_of_features, predictions_prob,
                            #                                             targeted_class_series)

                            result_performance_tuple, class_1_name, class_2_name = \
                                self._calculate_performance(classifier_type_name, num_of_features,
                                                            dataframe_column_names,
                                                            targeted_class_num_recods_tuples,
                                                            test_set_targeted_class_series,
                                                            predictions, predictions_prob)

                            end_time = time.time()
                            experiment_time = end_time - begin_time

                            result_performance_tuple = result_performance_tuple + (iteration, experiment_time)

                            result_performance_tuples.append(result_performance_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                      '#{}'.format(class_2_name), 'Classifier',
                                                      'Num_of_Features', 'Correctly',
                                                      "Incorrectly", 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                      'TN', 'FP', 'FN', 'TP',
                                                      'Selected_Features', 'iteration', 'time'])

        performance_result_df.to_csv(
            self._full_path_model_directory + "Classifier_Performance_Footprint_Features_claims_ordered_by_posts_no_cross_validation.csv")

        print("Done!!!!")

    def performance_function_training_size_random_choosing_no_cross_validation(self):
        author_feature_df = self._get_author_features_dataframe()
        original_labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                          author_feature_df)

        result_performance_tuples = []
        for class_num_of_records_dict in self._divide_records_by_targeted_classes_experiments:
            for iteration in range(1, self._iterations + 1):

                # for targeted_class_, num_of_records in class_num_of_records_dict.iteritems():
                #     targeted_class_num_recods_tuple = (targeted_class_, self._targeted_classes_dict[targeted_class_], num_of_records)
                #     targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuples = []
                keys = list(class_num_of_records_dict.keys())
                values = list(class_num_of_records_dict.values())
                print(("#records = {0}-{1}, iteration = {2}".format(values[0], values[1], iteration)))
                targeted_class_num_recods_tuple = (keys[0], self._targeted_classes_dict[keys[0]], values[0])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuple = (keys[1], self._targeted_classes_dict[keys[1]], values[1])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                training_set_df, test_set_df = self._create_training_and_test_df_by_num_of_records_dict(
                    class_num_of_records_dict,
                    original_labeled_features_df)

                training_set_df, training_set_targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
                    training_set_df)
                training_set_df = training_set_df.convert_objects(convert_numeric=True)
                training_set_df = training_set_df.fillna(0)

                test_set_df, test_set_targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
                    test_set_df)
                test_set_df = test_set_df.convert_objects(convert_numeric=True)
                test_set_df = test_set_df.fillna(0)

                for num_of_features in self._num_of_features_to_train:
                    targeted_training_set_df, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                        training_set_df, training_set_targeted_class_series, num_of_features)

                    targeted_test_set_df = pd.DataFrame(test_set_df, columns=dataframe_column_names)
                    dataframe_column_names = '||'.join(dataframe_column_names)
                    for classifier_type_name in self._classifier_type_names:
                        begin_time = time.time()

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)

                        if selected_classifier is not None:

                            if classifier_type_name == "XGBoost":
                                targeted_training_set_df = self._set_dataframe_columns_types(targeted_training_set_df)

                            selected_classifier.fit(targeted_training_set_df, training_set_targeted_class_series)

                            predictions = selected_classifier.predict(targeted_test_set_df)
                            predictions_prob = selected_classifier.predict_proba(targeted_test_set_df)

                            # if self._export_file_with_predictions_during_training:
                            #     self._export_confidence_performance_with_tweets(index_field_series, classifier_type_name,
                            #                                             num_of_features, predictions_prob,
                            #                                             targeted_class_series)

                            result_performance_tuple, class_1_name, class_2_name = \
                                self._calculate_performance(classifier_type_name, num_of_features,
                                                            dataframe_column_names,
                                                            targeted_class_num_recods_tuples,
                                                            test_set_targeted_class_series,
                                                            predictions, predictions_prob)

                            end_time = time.time()
                            experiment_time = end_time - begin_time

                            result_performance_tuple = result_performance_tuple + (iteration, experiment_time)

                            result_performance_tuples.append(result_performance_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                      '#{}'.format(class_2_name), 'Classifier',
                                                      'Num_of_Features', 'Correctly',
                                                      "Incorrectly", 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                      'TN', 'FP', 'FN', 'TP',
                                                      'Selected_Features', 'iteration', 'time'])

        performance_result_df.to_csv(
            self._full_path_model_directory + "Classifier_Performance_Footprint_features_Train_Size_Random_no_cross_validation.csv")

        print("Done!!!!")

    def performance_function_of_training_size_ordered_by_number_of_posts_cross_validation(self):
        author_feature_df = self._get_author_features_dataframe()
        original_labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                          author_feature_df)

        claim_tuples = self._db.get_claims_tuples()
        claims_df = pd.DataFrame(claim_tuples, columns=["claim_id", "title", "description", "url", "verdict_date",
                                                        "keywords", "domain", "verdict", "category", "claim_topic",
                                                        "claim_post_id", "claim_ext_id", "sub_category",
                                                        "main_category",
                                                        "secondary_category"])

        claim_ordered_by_num_of_posts_tuples = self._db.get_claim_ordered_by_num_of_posts()
        claim_ordered_by_num_of_posts_df = pd.DataFrame(claim_ordered_by_num_of_posts_tuples,
                                                        columns=["claim_id", "num_of_posts"])

        claims_df = self._replace_nominal_values_to_numeric(claims_df)

        real_claims_df = claims_df[claims_df['verdict'] == 0]
        fake_claims_df = claims_df[claims_df['verdict'] == 1]

        real_claim_ordered_by_num_of_posts_df = pd.merge(claim_ordered_by_num_of_posts_df, real_claims_df,
                                                         on=['claim_id'], how='inner')

        fake_claim_ordered_by_num_of_posts_df = pd.merge(claim_ordered_by_num_of_posts_df, fake_claims_df,
                                                         on=['claim_id'], how='inner')

        result_performance_tuples = []
        for class_num_of_records_dict in self._divide_records_by_targeted_classes_experiments:
            for iteration in range(1, self._iterations + 1):

                targeted_class_num_recods_tuples = []
                keys = list(class_num_of_records_dict.keys())
                values = list(class_num_of_records_dict.values())
                print(("#records = {0}-{1}, iteration = {2}".format(values[0], values[1], iteration)))
                targeted_class_num_recods_tuple = (keys[0], self._targeted_classes_dict[keys[0]], values[0])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuple = (keys[1], self._targeted_classes_dict[keys[1]], values[1])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                labeled_features_df = self._create_training_df_by_real_and_fake_claims_ordered_by_posts_dict(
                    class_num_of_records_dict, original_labeled_features_df, real_claim_ordered_by_num_of_posts_df,
                    fake_claim_ordered_by_num_of_posts_df)

                # labeled_features_df = self._create_training_df_by_num_of_records_dict(class_num_of_records_dict,
                #                                                                       original_labeled_features_df)

                labeled_features_df, targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
                    labeled_features_df)

                labeled_features_df = labeled_features_df.convert_objects(convert_numeric=True)
                labeled_features_df = labeled_features_df.fillna(0)
                # labeled_features_dataframe = self._set_dataframe_columns_types(labeled_features_dataframe)

                for num_of_features in self._num_of_features_to_train:
                    targeted_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                        labeled_features_df, targeted_class_series, num_of_features)

                    dataframe_column_names = '||'.join(dataframe_column_names)

                    for classifier_type_name in self._classifier_type_names:
                        begin_time = time.time()

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)

                        if selected_classifier is not None:

                            if classifier_type_name == "XGBoost":
                                targeted_dataframe = self._set_dataframe_columns_types(targeted_dataframe)

                            predictions = cross_val_predict(selected_classifier, targeted_dataframe,
                                                            targeted_class_series,
                                                            cv=self._k_for_fold)
                            predictions_prob = cross_val_predict(selected_classifier, targeted_dataframe,
                                                                 targeted_class_series,
                                                                 cv=self._k_for_fold, method='predict_proba')

                            if self._export_file_with_predictions_during_training:
                                self._export_confidence_performance_with_tweets(index_field_series,
                                                                                classifier_type_name,
                                                                                num_of_features, predictions_prob,
                                                                                targeted_class_series)

                            result_performance_tuple, class_1_name, class_2_name = \
                                self._calculate_performance(classifier_type_name, num_of_features,
                                                            dataframe_column_names,
                                                            targeted_class_num_recods_tuples, targeted_class_series,
                                                            predictions, predictions_prob)

                            end_time = time.time()
                            experiment_time = end_time - begin_time

                            result_performance_tuple = result_performance_tuple + (iteration, experiment_time)

                            result_performance_tuples.append(result_performance_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                      '#{}'.format(class_2_name), 'Classifier',
                                                      'Num_of_Features', 'Correctly',
                                                      "Incorrectly", 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                      'TN', 'FP', 'FN', 'TP',
                                                      'Selected_Features', 'iteration', 'time'])

        performance_result_df.to_csv(
            self._full_path_model_directory + "Classifier_Performance_Info_Gain_Greater_Than_zero_Features_claims_ordered_by_num_posts_cross_validation.csv")

        print("Done!!!!")

    # 1. Divide to records by targeted class
    # 2. Train classifiers and make iterations
    def performance_function_training_size_random_cross_validation_experiments(self):
        author_feature_df = self._get_author_features_dataframe()
        original_labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                          author_feature_df)

        if self._is_create_equal_number_of_records_per_class:
            original_labeled_features_df = self._create_equal_number_of_records_per_class(original_labeled_features_df)

        # for targeted_class, num in self._targeted_classes_dict.iteritems():
        #     targeted_class_df = original_labeled_features_df[
        #         original_labeled_features_df[self._targeted_class_field_name] == targeted_class]
        #     num_of_records = len(targeted_class_df.index)
        #
        #     targeted_class_num_recods_tuple = (targeted_class, num, num_of_records)
        #     targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

        result_performance_tuples = []
        for class_num_of_records_dict in self._divide_records_by_targeted_classes_experiments:
            for iteration in range(1, self._iterations + 1):

                # for targeted_class_, num_of_records in class_num_of_records_dict.iteritems():
                #     targeted_class_num_recods_tuple = (targeted_class_, self._targeted_classes_dict[targeted_class_], num_of_records)
                #     targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuples = []
                keys = list(class_num_of_records_dict.keys())
                values = list(class_num_of_records_dict.values())
                print(("#records = {0}-{1}, iteration = {2}".format(values[0], values[1], iteration)))
                targeted_class_num_recods_tuple = (keys[0], self._targeted_classes_dict[keys[0]], values[0])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                targeted_class_num_recods_tuple = (keys[1], self._targeted_classes_dict[keys[1]], values[1])
                targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

                labeled_features_df = self._create_training_df_by_num_of_records_dict(class_num_of_records_dict,
                                                                                      original_labeled_features_df)

                labeled_features_df, targeted_class_series, index_field_series = self._prepare_dataframe_for_learning(
                    labeled_features_df)

                labeled_features_df = labeled_features_df.convert_objects(convert_numeric=True)
                labeled_features_df = labeled_features_df.fillna(0)
                # labeled_features_dataframe = self._set_dataframe_columns_types(labeled_features_dataframe)

                for num_of_features in self._num_of_features_to_train:
                    targeted_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                        labeled_features_df, targeted_class_series, num_of_features)

                    dataframe_column_names = '||'.join(dataframe_column_names)

                    for classifier_type_name in self._classifier_type_names:
                        begin_time = time.time()

                        selected_classifier = self._select_classifier_by_type(classifier_type_name)

                        if selected_classifier is not None:

                            if classifier_type_name == "XGBoost":
                                targeted_dataframe = self._set_dataframe_columns_types(targeted_dataframe)

                            predictions = cross_val_predict(selected_classifier, targeted_dataframe,
                                                            targeted_class_series,
                                                            cv=self._k_for_fold)
                            predictions_prob = cross_val_predict(selected_classifier, targeted_dataframe,
                                                                 targeted_class_series,
                                                                 cv=self._k_for_fold, method='predict_proba')

                            if self._export_file_with_predictions_during_training:
                                self._export_confidence_performance_with_tweets(index_field_series,
                                                                                classifier_type_name,
                                                                                num_of_features, predictions_prob,
                                                                                targeted_class_series)

                            result_performance_tuple, class_1_name, class_2_name = \
                                self._calculate_performance(classifier_type_name, num_of_features,
                                                            dataframe_column_names,
                                                            targeted_class_num_recods_tuples, targeted_class_series,
                                                            predictions, predictions_prob)

                            end_time = time.time()
                            experiment_time = end_time - begin_time

                            result_performance_tuple = result_performance_tuple + (iteration, experiment_time)

                            result_performance_tuples.append(result_performance_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                      '#{}'.format(class_2_name), 'Classifier',
                                                      'Num_of_Features', 'Correctly',
                                                      "Incorrectly", 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                      'TN', 'FP', 'FN', 'TP',
                                                      'Selected_Features', 'iteration', 'time'])

        performance_result_df.to_csv(
            self._full_path_model_directory + "Classifier_Performance_Info_Gain_Greater_Than_zero_Features_Random_Cross_Validation.csv")

        print("Done!!!!")

    def load_trained_classifier_and_predict(self):
        result_performance_tuples = []
        full_path_selected_model = self._full_path_model_directory + self._prepared_classifier_file_name
        selected_features_path = self._full_path_model_directory + self._prepared_classifier_selected_features_file_name

        trained_classifier = joblib.load(full_path_selected_model)
        selected_features = joblib.load(selected_features_path)

        author_features_dataframe = self._get_author_features_dataframe()
        labeled_features_dataframe = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                        author_features_dataframe)

        labeled_features_dataframe, labeled_targeted_class_series, index_field_series = \
            self._prepare_dataframe_for_learning(labeled_features_dataframe)

        original_column_names = list(labeled_features_dataframe.columns.values)
        features_to_remove = self._calculate_features_to_remove(selected_features, original_column_names)
        labeled_features_dataframe = self._remove_features(features_to_remove, labeled_features_dataframe)

        # if best_classifier_name == "XGBoost":
        #     labeled_features_dataframe = self._set_dataframe_columns_types(labeled_features_dataframe)

        predictions_series, predictions_proba_series, predictions_proba, predictions_series_int = \
            self._predict_classifier(trained_classifier, labeled_features_dataframe)

        result_performance_tuple, class_1_name, class_2_name = self._calculate_performance(
            self._targeted_classifier_name, self._targeted_classifier_num_of_features,
            selected_features, [(-1, -1, -1), (-1, -1, -1)], labeled_targeted_class_series, predictions_series_int,
            predictions_proba)

        result_performance_tuples.append(result_performance_tuple)
        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                      '#{}'.format(class_2_name), 'Classifier',
                                                      'Num_of_Features', '#Corrected',
                                                      "Incorrected", 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                      'TN', 'FP', 'FN', 'TP',
                                                      'Selected_Features'])

        performance_result_df.to_csv(self._full_path_model_directory + "Classifier_Performance_Results.csv")

        df = pd.DataFrame(index_field_series, columns=[self._index_field])

        df["Best_Classifier"] = self._targeted_classifier_name
        df["Num_of_Features"] = df.shape[0] * [self._targeted_classifier_num_of_features]
        # df["Best_Features"] = df.shape[0] * best_performance_feature_names

        df["confidence"] = predictions_proba_series.values
        df['actual'] = labeled_targeted_class_series

        claim_tweet_connections = self._db.get_claim_tweet_connections()

        df_claim_tweet_connections = pd.DataFrame(claim_tweet_connections, columns=['claim_id', 'post_id'])

        claims_group_by_posts_df = df_claim_tweet_connections.groupby(['claim_id']).agg(['count'])

        df["Num_of_posts"] = claims_group_by_posts_df

        df.to_csv(self._full_path_model_directory + "labeled_predictions.csv")

        print("Done!!")

    def perform_transfer_learning_using_models(self):
        author_features_dataframe = self._get_author_features_dataframe()
        labeled_features_dataframe = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                        author_features_dataframe)

        if self._is_create_equal_number_of_records_per_class:
            labeled_features_dataframe = self._create_equal_number_of_records_per_class(labeled_features_dataframe)

        targeted_class_num_recods_tuples = []
        for targeted_class, num in self._targeted_classes_dict.items():
            targeted_class_df = labeled_features_dataframe[
                labeled_features_dataframe[self._targeted_class_field_name] == targeted_class]
            num_of_records = len(targeted_class_df.index)

            targeted_class_num_recods_tuple = (targeted_class, num, num_of_records)
            targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

        labeled_features_dataframe, targeted_class_series, index_field_series = \
            self._prepare_dataframe_for_learning(labeled_features_dataframe)

        labeled_features_dataframe = labeled_features_dataframe.convert_objects(convert_numeric=True)
        labeled_features_dataframe = labeled_features_dataframe.fillna(0)

        X = labeled_features_dataframe
        y = targeted_class_series

        features = ["LDATopicFeatureGenerator_topic_skewness","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_median_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_min_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_kurtosis_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_posts_age_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_skew_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_skew_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_skew_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_min_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_kurtosis_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_skew_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_median_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_skew_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_std_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_mean_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_kurtosis_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_skew_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_sum_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_max_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_min_0.8_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_sum","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_median","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_mean","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_max","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_min","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_kurtosis","AggregatedAuthorsPostsFeatureGenerator_posts_age_sum","AggregatedAuthorsPostsFeatureGenerator_posts_age_std","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_sum","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_std","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_max","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_skew","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_std","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_max","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_min","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_skew","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_std","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_max","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_std","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_max","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_kurtosis","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_skew","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_sum","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_std","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_median","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_max","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_sum","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_std","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_max","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_skew","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_sum","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_std","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_mean","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_max","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_kurtosis","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_skew","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_sum","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_max","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_min","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_sum_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_median_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_min_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_kurtosis_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_posts_age_sum_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_sum_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_skew_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_min_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_median_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_kurtosis_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_skew_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_kurtosis_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_skew_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_sum_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_median_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_min_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_skew_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_sum_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_sum_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_std_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_mean_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_kurtosis_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_skew_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_max_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_min_0.6_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_sum_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_median_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_min_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_kurtosis_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_posts_age_sum_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_sum_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_skew_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_min_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_skew_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_min_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_kurtosis_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_skew_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_skew_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_sum_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_median_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_skew_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_sum_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_skew_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_sum_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_std_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_mean_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_kurtosis_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_max_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_min_0.4_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_sum_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_median_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_min_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_authors_age_diff_kurtosis_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_posts_age_sum_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_sum_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_followers_skew_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_friends_skew_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_kurtosis_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_statuses_skew_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_favorites_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_sum_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_listed_count_skew_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_sum_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_protected_skew_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_sum_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_max_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_num_of_verified_skew_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_std_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_mean_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_min_0.2_newest","AggregatedAuthorsPostsFeatureGenerator_friends_followers_ratio_skew_0.2_newest","BehaviorFeatureGenerator_retweet_count","SyntaxFeatureGenerator_average_links","SyntaxFeatureGenerator_average_user_mentions","SyntaxFeatureGenerator_average_post_lenth","Sentiment_Feature_Generator_authors_posts_semantic_compound_median","Sentiment_Feature_Generator_authors_posts_semantic_compound_max","Sentiment_Feature_Generator_authors_posts_semantic_positive_std","Sentiment_Feature_Generator_authors_posts_semantic_positive_mean","Sentiment_Feature_Generator_authors_posts_semantic_negative_std","Sentiment_Feature_Generator_authors_posts_semantic_negative_median","Sentiment_Feature_Generator_authors_posts_semantic_negative_mean","Sentiment_Feature_Generator_authors_posts_semantic_negative_max","Sentiment_Feature_Generator_authors_posts_semantic_negative_min","Sentiment_Feature_Generator_authors_posts_semantic_neutral_std","Sentiment_Feature_Generator_authors_posts_semantic_neutral_median","Sentiment_Feature_Generator_authors_posts_semantic_neutral_mean","Sentiment_Feature_Generator_authors_posts_semantic_neutral_max","Sentiment_Feature_Generator_authors_posts_semantic_neutral_min","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d14","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d17","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d25","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d26","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d32","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d34","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d43","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d48","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d89","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d90","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d97","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d106","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d111","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d123","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d124","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d128","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d130","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d131","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d132","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d139","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d141","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d150","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d152","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d154","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d156","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d164","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d166","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d171","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d179","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d186","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d203","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d217","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d220","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d234","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d242","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d252","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d256","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d261","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d272","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d275","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d277","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d278","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d284","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d287","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d289","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d292","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d293","GloveWordEmbeddingsFeatureGenerator_max_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d295","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d9","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d10","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d21","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d26","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d32","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d35","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d37","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d43","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d65","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d68","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d74","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d76","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d82","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d87","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d101","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d113","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d114","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d124","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d136","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d144","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d157","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d159","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d168","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d169","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d171","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d178","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d181","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d184","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d188","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d191","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d211","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d212","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d213","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d236","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d237","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d238","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d240","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d244","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d257","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d265","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d266","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d276","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d281","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d288","GloveWordEmbeddingsFeatureGenerator_min_claims_content_author_word_embeddings_glove_wikipedia_model_300d_300_dim_d299","Text_Anlalyser_Feature_Generator_num_of_chars_on_avg","Text_Anlalyser_Feature_Generator_num_of_adjectives_on_avg","Text_Anlalyser_Feature_Generator_num_of_uppercase_words_in_post_on_avg","Text_Anlalyser_Feature_Generator_number_of_precent_of_uppercased_posts","Text_Anlalyser_Feature_Generator_num_of_formal_words_on_avg","Text_Anlalyser_Feature_Generator_precent_of_formal_words_on_avg","Text_Anlalyser_Feature_Generator_num_of_question_marks_on_avg","Text_Anlalyser_Feature_Generator_num_of_comas_on_avg","Text_Anlalyser_Feature_Generator_num_of_stop_words_on_avg","Text_Anlalyser_Feature_Generator_precent_of_stop_words_on_avg","TemporalFeatureGenerator_posts_temporal_kurtosis_delta_time_1","TemporalFeatureGenerator_posts_temporal_skew_delta_time_1","TemporalFeatureGenerator_authors_temporal_sum_delta_time_1","TemporalFeatureGenerator_authors_temporal_std_delta_time_1","TemporalFeatureGenerator_authors_temporal_median_delta_time_1","TemporalFeatureGenerator_authors_temporal_mean_delta_time_1","TemporalFeatureGenerator_authors_temporal_max_delta_time_1","TemporalFeatureGenerator_authors_temporal_sum_delta_time_30","TemporalFeatureGenerator_authors_temporal_std_delta_time_30","TemporalFeatureGenerator_authors_temporal_median_delta_time_30","TemporalFeatureGenerator_authors_temporal_mean_delta_time_30","TemporalFeatureGenerator_authors_temporal_max_delta_time_30","TemporalFeatureGenerator_authors_temporal_min_delta_time_30","TemporalFeatureGenerator_authors_temporal_kurtosis_delta_time_30","TemporalFeatureGenerator_authors_temporal_skew_delta_time_30","TemporalFeatureGenerator_authors_temporal_sum_delta_time_365","TemporalFeatureGenerator_authors_temporal_std_delta_time_365","TemporalFeatureGenerator_authors_temporal_median_delta_time_365","TemporalFeatureGenerator_authors_temporal_mean_delta_time_365","TemporalFeatureGenerator_authors_temporal_max_delta_time_365","TemporalFeatureGenerator_authors_temporal_min_delta_time_365","TemporalFeatureGenerator_authors_temporal_kurtosis_delta_time_365","TemporalFeatureGenerator_authors_temporal_skew_delta_time_365","TF_IDF_Feature_Generator_min","TF_IDF_Feature_Generator_max","TF_IDF_Feature_Generator_median","TF_IDF_Feature_Generator_skew","TF_IDF_Feature_Generator_kurtosis"]

        with open('data/output/Experimentor/snopes_386_claims_k_fold/models/RandomForest_model_370_features.model') as classifier_file:
            pipe = pickle.load(classifier_file)
            predictions_prob = pipe.predict_proba(X[features])
            predictions = pipe.predict(X[features])
            result_performance_tuple1, class_1_name, class_2_name = \
                self._calculate_performance('RandomForest_model_370_features', 386, features,
                                            targeted_class_num_recods_tuples, y,
                                            predictions, predictions_prob)

            if self._export_file_with_predictions_during_training:
                self._export_confidence_performance_with_tweets(index_field_series, 'RandomForest_model_370_features',
                                                                len(features), predictions_prob,
                                                                targeted_class_series)

            prediction_confidence = pd.DataFrame()
            prediction_confidence['claim_id'] = X.index
            prediction_confidence['confidence'] = np.asarray(predictions_prob)[:, 1]
            prediction_confidence['prediction'] = predictions
            prediction_confidence.to_csv(self._full_path_model_directory + 'prediction_confidence.csv', index=False)

            X_test = X.tail(int(len(y) * (1 - self._training_percent)))
            y_test = y.tail(int(len(y) * (1 - self._training_percent)))

            result_performance_tuple2, class_1_name, class_2_name = \
                self._calculate_performance('RandomForest_test_data_only', 386, features,
                                            targeted_class_num_recods_tuples, y_test,
                                            pipe.predict(X_test[features]), pipe.predict_proba(X_test[features]))

            performance_result_df = pd.DataFrame([result_performance_tuple1, result_performance_tuple2],
                                                 columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                          '#{}'.format(class_2_name), 'Classifier',
                                                          'Num_of_Features', '#Corrected',
                                                          "Incorrected", 'AUC', 'Accuracy',
                                                          'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                          'TN', 'FP', 'FN', 'TP',
                                                          'Selected_Features'])
            performance_result_df.to_csv(self._full_path_model_directory + "snopes_model_prediction_on_antisemitic.csv")
            pass


    def perform_k_fold_cross_validation_and_predict_updated(self):
        author_features_dataframe = self._get_author_features_dataframe()
        labeled_features_dataframe = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                        author_features_dataframe)

        if self._is_create_equal_number_of_records_per_class:
            labeled_features_dataframe = self._create_equal_number_of_records_per_class(labeled_features_dataframe)

        targeted_class_num_recods_tuples = []
        for targeted_class, num in self._targeted_classes_dict.items():
            targeted_class_df = labeled_features_dataframe[
                labeled_features_dataframe[self._targeted_class_field_name] == targeted_class]
            num_of_records = len(targeted_class_df.index)

            targeted_class_num_recods_tuple = (targeted_class, num, num_of_records)
            targeted_class_num_recods_tuples.append(targeted_class_num_recods_tuple)

        labeled_features_dataframe.to_csv(self._full_path_model_directory + "Fake_News_Dataset.csv")
        labeled_features_dataframe, targeted_class_series, index_field_series = \
            self._prepare_dataframe_for_learning(labeled_features_dataframe)

        labeled_features_dataframe = labeled_features_dataframe.astype(float)
        labeled_features_dataframe = labeled_features_dataframe.fillna(0)

        # labeled_features_dataframe = self._set_dataframe_columns_types(labeled_features_dataframe)

        result_performance_tuples = []
        models_path = os.path.join(self._full_path_model_directory, 'models/')
        if not os.path.isdir(models_path):
            os.makedirs(models_path)

        for num_of_features in self._num_of_features_to_train:
            targeted_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
                labeled_features_dataframe, targeted_class_series, num_of_features)

            dataframe_column_names = '||'.join(dataframe_column_names)

            for classifier_type_name in self._classifier_type_names:
                selected_classifier = self._select_classifier_by_type(classifier_type_name)

                if selected_classifier is not None:

                    if classifier_type_name == "XGBoost":
                        targeted_dataframe = self._set_dataframe_columns_types(targeted_dataframe)

                    predictions = cross_val_predict(selected_classifier, targeted_dataframe, targeted_class_series,
                                                    cv=self._k_for_fold)
                    predictions_prob = cross_val_predict(selected_classifier, targeted_dataframe, targeted_class_series,
                                                         cv=self._k_for_fold, method='predict_proba')

                    clf_path = os.path.join(models_path,
                                            '{}_model_{}_features.model'.format(classifier_type_name, num_of_features))
                    with open(clf_path, 'wb') as clf_file:
                        selected_classifier.fit(targeted_dataframe, targeted_class_series)
                        pickle.dump(selected_classifier, clf_file)

                    if self._export_file_with_predictions_during_training:
                        self._export_confidence_performance_with_tweets(index_field_series, classifier_type_name,
                                                                        num_of_features, predictions_prob,
                                                                        targeted_class_series)

                    result_performance_tuple, class_1_name, class_2_name = \
                        self._calculate_performance(classifier_type_name, num_of_features, dataframe_column_names,
                                                    targeted_class_num_recods_tuples, targeted_class_series,
                                                    predictions, predictions_prob)

                    result_performance_tuples.append(result_performance_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples,
                                             columns=['Target_Class_Name', '#{}'.format(class_1_name),
                                                      '#{}'.format(class_2_name), 'Classifier',
                                                      'Num_of_Features', '#Corrected',
                                                      "Incorrected", 'AUC', 'Accuracy',
                                                      'F1', 'Precision', 'Recall', 'Confusion Matrix',
                                                      'TN', 'FP', 'FN', 'TP',
                                                      'Selected_Features'])

        performance_result_df.to_csv(
            self._full_path_model_directory + "Classifier_Performance_Results_370_claims_only_links_5_top_features.csv")

        # self._find_best_classifier_and_predict_on_unlabeled(performance_result_df, labeled_features_dataframe,
        #                                                unlabeled_features_dataframe, targeted_class_series,
        #                                                unlabeled_targeted_class_series, unlabeled_index_field_series)

        # self._find_best_classifier_and_predict_on_labeled(performance_result_df, labeled_features_dataframe,
        #                                                   targeted_class_series, index_field_series)

    def _get_author_features_dataframe(self):
        start_time = timeit.default_timer()
        print((
                "_get_author_features_dataframe started for " + self.__class__.__name__ + " started at " + str(
            start_time)))
        data_frame_creator = DataFrameCreator(self._db)
        data_frame_creator.create_lazy_author_features_df()
        author_features_dataframe = data_frame_creator.get_author_features_data_frame()

        end_time = timeit.default_timer()
        print(("_get_author_features_dataframe ended for " + self.__class__.__name__ + " ended at " + str(end_time)))
        print(('author features df load time: {} sec'.format(str(end_time - start_time))))
        return author_features_dataframe

    def _retreive_unlabeled_authors_dataframe(self, dataframe):
        # unlabeled_data_frame = dataframe[dataframe.author_type.isnull()]
        unlabeled_data_frame = dataframe.loc[dataframe[self._targeted_class_name].isnull()]
        return unlabeled_data_frame

    def _prepare_dataframe_for_learning(self, dataframe):
        start_time = time.time()
        print((
                "_prepare_dataframe_for_learning started for " + self.__class__.__name__ + " started at " + str(
            start_time)))

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

        # dataframe = replace_nominal_class_to_numeric(dataframe, self._optional_classes)
        dataframe = self._replace_nominal_values_to_numeric(dataframe)

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
            # targeted_dataframe.to_csv(self._full_path_model_directory + "Fake_News_Dataset.csv")

        else:
            targeted_dataframe = dataframe
        return targeted_dataframe, targeted_class_series, index_field_series

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
        print((', '.join(reduced_dataframe_column_names)))

        reduced_dataframe = pd.DataFrame(k_best_features, columns=reduced_dataframe_column_names)

        return reduced_dataframe, reduced_dataframe_column_names

    def _select_classifier_by_type(self, classifier_type_name):
        selected_classifier = None

        if classifier_type_name == Classifiers.RandomForest:
            # selected_classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_features=0.1)
            selected_classifier = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=10)

        elif classifier_type_name == Classifiers.DecisionTree:
            selected_classifier = tree.DecisionTreeClassifier()

        elif classifier_type_name == Classifiers.AdaBoost:
            # selected_classifier = sklearn.ensemble.AdaBoostClassifier(n_estimators=30)
            selected_classifier = sklearn.ensemble.AdaBoostClassifier(n_estimators=50)

        elif classifier_type_name == Classifiers.XGBoost:
            selected_classifier = xgb.XGBClassifier(n_jobs=-1)

        return selected_classifier

    def _get_k_best_feature_names(self, k_best_classifier, original_dataframe):
        mask = k_best_classifier.get_support()
        best_feature_names = []
        column_names = list(original_dataframe.columns.values)
        for boolean_value, feature_name in zip(mask, column_names):
            if boolean_value == True:
                best_feature_names.append(feature_name)
        return best_feature_names

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

    # def _calculate_performance(self, test_class, total_confidence_predictions, total_binary_predictions,
    #                            selected_features):
    #     try:
    #         auc_score = roc_auc_score(test_class, total_confidence_predictions)
    #     except:
    #         auc_score = -1
    #     accuracy = accuracy_score(test_class, total_binary_predictions)
    #     f1 = f1_score(test_class, total_binary_predictions)
    #     precision = precision_score(test_class, total_binary_predictions)
    #     recall = recall_score(test_class, total_binary_predictions)
    #
    #     result_performance_tuple = (selected_features, auc_score, accuracy, f1, precision, recall)
    #     return result_performance_tuple

    def _calculate_features_to_remove(self, combination, feature_names):
        combination_set = set(combination)
        feature_names_set = set(feature_names)
        features_to_remove_set = feature_names_set - combination_set
        features_to_remove = list(features_to_remove_set)
        return features_to_remove

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

    def _replace_nominal_values_to_numeric(self, df):
        for class_value, num in self._targeted_classes_dict.items():
            df = df.replace(to_replace=class_value, value=num)
        return df

    def _create_equal_number_of_records_per_class(self, df):
        targeted_class_num_of_records_dict = {}
        targeted_class_num_of_records_tuples = []
        targeted_class_num_records_dfs = []
        for targeted_class, num in self._targeted_classes_dict.items():
            targeted_class_df = df[df[self._targeted_class_field_name] == targeted_class]
            targeted_class_num_records_dfs.append(targeted_class_df)
            num_of_records = len(targeted_class_df.index)
            targeted_class_num_of_records_dict[targeted_class] = num_of_records

            tuple_records = (targeted_class, num_of_records)
            targeted_class_num_of_records_tuples.append(tuple_records)

        original_records_df = pd.DataFrame(targeted_class_num_of_records_tuples, columns=['targeted_class', '#records'])
        original_records_df.to_csv(self._full_path_model_directory + "original_records.csv")

        num_of_records = list(targeted_class_num_of_records_dict.values())

        min_num_of_claims_per_class = min(num_of_records)

        dfs = []
        for targeted_class_num_records_df in targeted_class_num_records_dfs:
            sampled_df = targeted_class_num_records_df.sample(n=min_num_of_claims_per_class)
            dfs.append(sampled_df)

        training_df = pd.concat(dfs)

        shuffeled_training_df = training_df.sample(frac=1)
        return shuffeled_training_df

    def _divide_by_percentage(self, df):
        if len(self._classes_percent_dict) == 0:
            return df

        sample_targeted_class_dfs = []
        for class_num, percent in self._classes_percent_dict.items():
            sample_targeted_class_df = df.sample(frac=percent)
            sample_targeted_class_dfs.append(sample_targeted_class_df)
        training_set_df = pd.concat(sample_targeted_class_dfs)

        return training_set_df

    def _predict_classifier(self, selected_classifier, unlabeled_author_dataframe):

        predictions = selected_classifier.predict(unlabeled_author_dataframe)
        predictions_series_int = pd.Series(predictions)

        predictions_series = self._replace_predictions_class_from_int_to_string(predictions_series_int)

        predictions_proba = selected_classifier.predict_proba(unlabeled_author_dataframe)
        predictions_proba_series = pd.Series(predictions_proba[:, 1])

        return predictions_series, predictions_proba_series, predictions_proba, predictions_series_int

    def _write_predictions_into_file(self, classifier_type_name, num_of_features,
                                     unlabeled_index_field_series, predictions_series, predictions_proba_series):

        unlabeled_dataframe_with_prediction = pd.DataFrame(unlabeled_index_field_series,
                                                           columns=[self._index_field])

        unlabeled_dataframe_with_prediction.reset_index(drop=True, inplace=True)
        unlabeled_dataframe_with_prediction["predicted"] = predictions_series
        unlabeled_dataframe_with_prediction["prediction"] = predictions_proba_series

        full_path = self._full_path_model_directory + "predictions_on_unlabeled_authors_" + self._targeted_class_name + "_" + \
                    classifier_type_name + "_" + str(num_of_features) + "_features.csv"
        # results_dataframe.to_csv(full_path)
        unlabeled_dataframe_with_prediction.to_csv(full_path, index=False)

        table_name = "unlabeled_predictions"
        self._db.drop_unlabeled_predictions(table_name)

        engine = self._db.engine
        unlabeled_dataframe_with_prediction.to_sql(name=table_name, con=engine)

    def _replace_predictions_class_from_int_to_string(self, predictions_series):
        predictions_series = self._replace_numeric_class_to_nominal(predictions_series)
        return predictions_series

    def _replace_numeric_class_to_nominal(self, predictions_series):
        for class_name, num in self._targeted_classes_dict.items():
            predictions_series = predictions_series.replace(to_replace=num, value=class_name)
        return predictions_series

    def _find_best_classifier_and_predict_on_unlabeled(self, performance_result_df, labeled_features_dataframe,
                                                       unlabeled_features_dataframe, targeted_class_series,
                                                       unlabeled_targeted_class_series, unlabeled_index_field_series):

        best_classifier, best_classifier_name, best_performance_num_of_features, best_performance_feature_names = \
            self._find_best_classifier(performance_result_df, labeled_features_dataframe, targeted_class_series)

        self._save_trained_model(best_classifier, best_classifier_name, len(best_performance_feature_names),
                                 best_performance_feature_names)

        # original_column_names = list(unlabeled_features_dataframe.columns.values)
        # features_to_remove = self._calculate_features_to_remove(best_performance_feature_names, original_column_names)
        # unlabeled_features_dataframe = self._remove_features(features_to_remove, unlabeled_features_dataframe)
        #
        #
        #
        # if best_classifier_name == "XGBoost":
        #     unlabeled_features_dataframe = self._set_dataframe_columns_types(unlabeled_features_dataframe)
        #
        #
        #
        #
        # predictions_series, predictions_proba_series = self._predict_classifier(best_classifier,
        #                                                                         unlabeled_features_dataframe)
        #
        # self._write_predictions_into_file(best_classifier_name, best_performance_num_of_features,
        #                                   unlabeled_index_field_series, predictions_series,
        #                                   predictions_proba_series)

    def _find_best_classifier(self, performance_result_df, labeled_features_dataframe, targeted_class_series):

        performance_max_auc = performance_result_df['AUC'].argmax()
        best_performance_df = performance_result_df.iloc[[performance_max_auc]]
        best_classifier_name = best_performance_df['Classifier'].tolist()[0]
        best_performance_feature_names = best_performance_df['Selected_Features'].tolist()[0]
        best_performance_num_of_features = best_performance_df['Num_of_Features'].tolist()[0]

        best_classifier = self._select_classifier_by_type(best_classifier_name)

        original_column_names = list(labeled_features_dataframe.columns.values)
        features_to_remove = self._calculate_features_to_remove(best_performance_feature_names, original_column_names)
        reduced_original_training_df = self._remove_features(features_to_remove, labeled_features_dataframe)

        if best_classifier_name == "XGBoost":
            reduced_original_training_df = self._set_dataframe_columns_types(reduced_original_training_df)

        best_classifier.fit(reduced_original_training_df, targeted_class_series)

        return best_classifier, best_classifier_name, best_performance_num_of_features, best_performance_feature_names

    def _find_best_classifier_and_predict_on_labeled(self, performance_result_df,
                                                     labeled_features_dataframe, targeted_class_series,
                                                     index_field_series):
        best_classifier, best_classifier_name, best_performance_num_of_features, best_performance_feature_names = \
            self._find_best_classifier(performance_result_df, labeled_features_dataframe, targeted_class_series)

        original_column_names = list(labeled_features_dataframe.columns.values)
        features_to_remove = self._calculate_features_to_remove(best_performance_feature_names, original_column_names)
        labeled_features_dataframe = self._remove_features(features_to_remove, labeled_features_dataframe)

        # labeled_features_dataframe, dataframe_column_names = self._reduce_dimensions_by_num_of_features(
        #     labeled_features_dataframe, targeted_class_series, best_performance_num_of_features)

        if best_classifier_name == "XGBoost":
            labeled_features_dataframe = self._set_dataframe_columns_types(labeled_features_dataframe)

        predictions_series, predictions_proba_series = self._predict_classifier(best_classifier,
                                                                                labeled_features_dataframe)
        df = pd.DataFrame(index_field_series, columns=[self._index_field])

        df["Best_Classifier"] = best_classifier_name
        df["Num_of_Features"] = df.shape[0] * [best_performance_num_of_features]
        # df["Best_Features"] = df.shape[0] * best_performance_feature_names

        df["confidence"] = predictions_proba_series.values
        df['actual'] = targeted_class_series

        claim_tweet_connections = self._db.get_claim_tweet_connections()

        df_claim_tweet_connections = pd.DataFrame(claim_tweet_connections, columns=['claim_id', 'post_id'])

        claims_group_by_posts_df = df_claim_tweet_connections.groupby(['claim_id']).agg(['count'])

        df["Num_of_posts"] = claims_group_by_posts_df

        df.to_csv(self._full_path_model_directory + "labeled_predictions.csv")

    def _set_dataframe_columns_types(self, df):
        column_names = df.columns.values
        for column_name in column_names:
            print(("feature_name: " + column_name))
            feature_series = df[column_name]
            feature_series = feature_series.astype(np.float64)
            # feature_series = feature_series.astype(np.int64)
            df[column_name] = feature_series
        return df

    def _calculate_performance(self, classifier_type_name, num_of_features, dataframe_column_names,
                               targeted_class_num_recods_tuples, targeted_class_series, predictions,
                               predictions_prob):
        try:
            # auc_score = roc_auc_score(targeted_class_series, prediction_probabilities)
            auc_score = roc_auc_score(targeted_class_series, np.asarray(predictions_prob)[:, 1])
        except:
            auc_score = -1
        accuracy = accuracy_score(targeted_class_series, predictions)
        f1 = f1_score(targeted_class_series, predictions)
        precision = precision_score(targeted_class_series, predictions)
        recall = recall_score(targeted_class_series, predictions)

        conf_matrix = confusion_matrix(targeted_class_series, predictions)
        tn, fp, fn, tp = conf_matrix.ravel()

        num_of_correct_instances, \
        num_of_incorrect_instances = calculate_correctly_and_not_correctly_instances(conf_matrix)

        class_1_name = targeted_class_num_recods_tuples[0][0]
        class_1_records = targeted_class_num_recods_tuples[0][2]

        class_2_name = targeted_class_num_recods_tuples[1][0]
        class_2_records = targeted_class_num_recods_tuples[1][2]

        result_performance_tuple = (self._targeted_class_name, class_1_records, class_2_records,
                                    classifier_type_name, num_of_features, num_of_correct_instances,
                                    num_of_incorrect_instances, auc_score, accuracy, f1,
                                    precision, recall, conf_matrix, tn, fp, fn, tp, dataframe_column_names)

        return result_performance_tuple, class_1_name, class_2_name

    def _export_confidence_performance_with_tweets(self, index_field_series, best_classifier_name,
                                                   best_performance_num_of_features, predictions_proba_series,
                                                   targeted_class_series):
        df = pd.DataFrame(index_field_series, columns=[self._index_field])

        df["Best_Classifier"] = best_classifier_name
        df["Num_of_Features"] = df.shape[0] * [best_performance_num_of_features]
        # df["Best_Features"] = df.shape[0] * best_performance_feature_names

        df["confidence"] = predictions_proba_series[:, 1]
        df['actual'] = targeted_class_series

        claim_tweet_connections = self._db.get_claim_tweet_connections()

        df_claim_tweet_connections = pd.DataFrame(claim_tweet_connections, columns=['claim_id', 'post_id'])

        claims_group_by_posts_df = df_claim_tweet_connections.groupby(['claim_id']).agg(['count'])

        df["Num_of_posts"] = claims_group_by_posts_df

        df.to_csv(self._full_path_model_directory + "labeled_predictions_correlation_to_tweets.csv")

    def _create_training_df_by_num_of_records_dict(self, class_num_of_records_dict, labeled_features_dataframe):
        classes = list(class_num_of_records_dict.keys())
        random_records_dfs = []
        for target_class in classes:
            number_of_records = class_num_of_records_dict[target_class]
            targeted_class_df = labeled_features_dataframe[
                labeled_features_dataframe[self._targeted_class_field_name] == target_class]

            sampled_df = targeted_class_df.sample(n=number_of_records)
            random_records_dfs.append(sampled_df)

        training_df = pd.concat(random_records_dfs)

        shuffeled_training_df = training_df.sample(frac=1)
        return shuffeled_training_df

    def _create_training_df_by_real_and_fake_claims_ordered_by_posts_dict(self, class_num_of_records_dict,
                                                                          labeled_features_df,
                                                                          real_claim_ordered_by_num_of_posts_df,
                                                                          fake_claim_ordered_by_num_of_posts_df):
        classes = list(class_num_of_records_dict.keys())
        chosen_records_dfs = []
        for target_class in classes:
            number_of_records = class_num_of_records_dict[target_class]
            chosen_claim_ordered_by_posts_df = None
            if target_class == "True":
                chosen_claim_ordered_by_posts_df = real_claim_ordered_by_num_of_posts_df.head(number_of_records)
            elif target_class == "False":
                chosen_claim_ordered_by_posts_df = fake_claim_ordered_by_num_of_posts_df.head(number_of_records)
            chosen_claim_ids = chosen_claim_ordered_by_posts_df["claim_id"].tolist()

            sampled_df = labeled_features_df.ix[chosen_claim_ids]

            chosen_records_dfs.append(sampled_df)

        training_df = pd.concat(chosen_records_dfs)

        shuffeled_training_df = training_df.sample(frac=1)
        return shuffeled_training_df

    def _create_training_and_test_df_by_num_of_records_dict(self, class_num_of_records_dict,
                                                            labeled_features_dataframe):
        max_num_of_claims = 360
        classes = list(class_num_of_records_dict.keys())
        random_training_dfs = []
        random_test_dfs = []
        for target_class in classes:
            number_of_records = class_num_of_records_dict[target_class]
            test_set_number_of_records = (max_num_of_claims - number_of_records * 2) / 2
            targeted_class_df = labeled_features_dataframe[
                labeled_features_dataframe[self._targeted_class_field_name] == target_class]

            sampled_training_df = targeted_class_df.sample(n=number_of_records)
            random_training_dfs.append(sampled_training_df)

            rest_targeted_class_df = pd.concat([targeted_class_df, sampled_training_df]).drop_duplicates(keep=False)
            sampled_test_df = rest_targeted_class_df.sample(n=test_set_number_of_records)
            random_test_dfs.append(sampled_test_df)

        training_df = pd.concat(random_training_dfs)
        shuffeled_training_df = training_df.sample(frac=1)

        test_df = pd.concat(random_test_dfs)
        shuffeled_test_df = test_df.sample(frac=1)

        return shuffeled_training_df, shuffeled_test_df

    def _create_training_and_test_df_by_real_and_fake_claims_ordered_by_posts_dict(self, class_num_of_records_dict,
                                                                                   labeled_features_df,
                                                                                   real_claim_ordered_by_num_of_posts_df,
                                                                                   fake_claim_ordered_by_num_of_posts_df):

        max_num_of_claims = 360
        classes = list(class_num_of_records_dict.keys())
        selected_training_dfs = []
        selected_test_dfs = []
        for target_class in classes:
            training_set_number_of_records = class_num_of_records_dict[target_class]
            test_set_number_of_records = (max_num_of_claims - training_set_number_of_records * 2) / 2
            selected_top_claims_training_df = None
            selected_top_claims_test_df = None
            if target_class == "True":
                selected_top_claims_training_df, selected_top_claims_test_df = \
                    self._choose_claims_by_num_of_posts_to_training_and_test_set(training_set_number_of_records,
                                                                                 labeled_features_df,
                                                                                 test_set_number_of_records,
                                                                                 real_claim_ordered_by_num_of_posts_df)
            elif target_class == "False":
                selected_top_claims_training_df, selected_top_claims_test_df = \
                    self._choose_claims_by_num_of_posts_to_training_and_test_set(training_set_number_of_records,
                                                                                 labeled_features_df,
                                                                                 test_set_number_of_records,
                                                                                 fake_claim_ordered_by_num_of_posts_df)
            selected_training_dfs.append(selected_top_claims_training_df)
            selected_test_dfs.append(selected_top_claims_test_df)

        training_df = pd.concat(selected_training_dfs)
        shuffeled_training_df = training_df.sample(frac=1)

        test_df = pd.concat(selected_test_dfs)
        shuffeled_test_df = test_df.sample(frac=1)

        return shuffeled_training_df, shuffeled_test_df

    def _choose_claims_by_num_of_posts_to_training_and_test_set(self, training_set_number_of_records,
                                                                labeled_features_df,
                                                                test_set_number_of_records,
                                                                claim_ordered_by_num_of_posts_df):

        selected_top_claims_training_df = claim_ordered_by_num_of_posts_df.head(training_set_number_of_records)
        chosen_claim_ids = selected_top_claims_training_df["claim_id"].tolist()
        selected_top_claims_training_df = labeled_features_df.ix[chosen_claim_ids]

        rest_real_claim_ordered_by_num_of_posts_df = pd.concat(
            [claim_ordered_by_num_of_posts_df, selected_top_claims_training_df]).drop_duplicates(keep=False)
        selected_top_claims_test_df = rest_real_claim_ordered_by_num_of_posts_df.head(n=test_set_number_of_records)
        chosen_claim_ids = selected_top_claims_test_df["claim_id"].tolist()
        selected_top_claims_test_df = labeled_features_df.ix[chosen_claim_ids]

        return selected_top_claims_training_df, selected_top_claims_test_df

    def _create_train_and_test_dataframes_and_classes(self, targeted_dataframe, train_indexes,
                                                      test_indexes, targeted_class_series):
        train_set_dataframe = targeted_dataframe.loc[train_indexes.tolist()]
        test_set_dataframe = targeted_dataframe.loc[test_indexes.tolist()]
        train_class = targeted_class_series[train_indexes]
        test_class = targeted_class_series[test_indexes]
        return train_set_dataframe, test_set_dataframe, train_class, test_class

    def _calculate_performance_and_return_results(self, targeted_class_series, predictions, predictions_prob):
        try:
            # auc_score = roc_auc_score(targeted_class_series, prediction_probabilities)
            auc_score = roc_auc_score(targeted_class_series, predictions_prob[:, 1])
        except:
            auc_score = -1
        accuracy = accuracy_score(targeted_class_series, predictions)
        f1 = f1_score(targeted_class_series, predictions)
        precision = precision_score(targeted_class_series, predictions)
        recall = recall_score(targeted_class_series, predictions)

        return auc_score, accuracy, f1, precision, recall

    def _calculate_average_performance(self, total_auc, total_accuracy, total_f1, total_precision,
                                       total_recall):
        average_auc = sum(total_auc) / len(total_auc)
        average_accuracy = sum(total_accuracy) / len(total_accuracy)
        average_f1 = sum(total_f1) / len(total_f1)
        average_precision = sum(total_precision) / len(total_precision)
        average_recall = sum(total_recall) / len(total_recall)

        return average_auc, average_accuracy, average_f1, average_precision, average_recall

    def get_feature_importance_for_random_forest_classifier(self):
        author_feature_df = self._get_author_features_dataframe()
        labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                 author_feature_df)

        if self._replace_missing_values == 'zero':
            labeled_features_df = labeled_features_df.fillna(0)
        elif self._replace_missing_values == 'mean':
            labeled_features_df.fillna(labeled_features_df.mean(), inplace=True)

        labeled_features_df = self._replace_nominal_values_to_numeric(labeled_features_df)

        index_field_series = labeled_features_df.pop(self._index_field)
        targeted_class_series = labeled_features_df.pop(self._targeted_class_name)

        result_performance_tuples = []

        labeled_features_df = labeled_features_df.convert_objects(convert_numeric=True)

        X = labeled_features_df
        y = targeted_class_series

        # Split the data into 40% test and 60% training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        # Create a random forest classifier
        clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

        # Train the classifier
        clf.fit(X_train, y_train)
        df = self._calculate_feature_importance(clf, X_train)
        # df.to_csv()

    def _calculate_feature_importance(self, clf, X_train):
        feature_importances = []
        # # Print the name and gini importance of each feature
        for feature in zip(X_train.columns, clf.feature_importances_):
            feature_importances.append(feature)

        results_df = pd.DataFrame(feature_importances, columns=['Feature_Name', 'Gini Importance'])
        results_df = results_df.sort_values(by=['Gini Importance'], ascending=False)
        results_df.to_csv(self._full_path_model_directory + "feature_importance.csv")
        return results_df

    def perform_predict_by_date(self):
        author_feature_df = self._get_author_features_dataframe()
        labeled_features_df = retreive_labeled_authors_dataframe(self._targeted_class_name,
                                                                 author_feature_df)

        if self._replace_missing_values == 'zero':
            labeled_features_df = labeled_features_df.fillna(0)
        elif self._replace_missing_values == 'mean':
            labeled_features_df.fillna(labeled_features_df.mean(), inplace=True)

        labeled_features_df = self._replace_nominal_values_to_numeric(labeled_features_df)

        claim_tuples = self._db.get_claims_tuples()
        claim_id_claim_verdict_date_tuples = [(claim_tuple[0], claim_tuple[4]) for claim_tuple in claim_tuples]

        claim_id_claim_verdict_date_df = pd.DataFrame(claim_id_claim_verdict_date_tuples,
                                                      columns=['claim_id', 'verdict_date'])

        claim_id_claim_verdict_date_df['verdict_date'] = pd.to_datetime(claim_id_claim_verdict_date_df.verdict_date)
        united_df = labeled_features_df.merge(claim_id_claim_verdict_date_df, right_on='claim_id',
                                              left_on='ClaimFeatureGenerator_claim_id')

        sorted_united_df_with_date = united_df.sort_values(by='verdict_date')

        result_performance_tuples = []

        sorted_united_df = sorted_united_df_with_date.drop(['verdict_date'])
        sorted_united_df = sorted_united_df.convert_objects(convert_numeric=True)

        X = sorted_united_df
        # y = targeted_class_series

        X_train = X.head(int(len(sorted_united_df) * self._training_percent))
        # X_train = X_train.reset_index()
        X_test = X.tail(int(len(sorted_united_df) - len(X_train)))
        # X_test = X_test.reset_index()

        training_size = len(X_train)
        test_size = len(X_test)

        training_true_num = len(X_train[X_train[self._targeted_class_name] == 0])
        training_false_num = len(X_train[X_train[self._targeted_class_name] == 1])

        test_true_num = len(X_test[X_test[self._targeted_class_name] == 0])
        test_false_num = len(X_test[X_test[self._targeted_class_name] == 1])

        # index_field_series = sorted_united_df.pop(self._index_field)
        # verdict_date_series = sorted_united_df.pop('verdict_date')
        # targeted_class_series = sorted_united_df.pop(self._targeted_class_name)
        # y = targeted_class_series

        y_train = X_train.pop(self._targeted_class_name)
        training_verdict_date = X_train.pop('verdict_date')
        training_indexes = X_train.pop(self._index_field)
        training_indexes = X_train.pop('claim_id')

        y_test = X_test.pop(self._targeted_class_name)
        test_verdict_date = X_test.pop('verdict_date')
        test_indexes = X_test.pop(self._index_field)
        test_indexes = X_test.pop('claim_id')

        cols = X_train.columns

        # classifier = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
        max_depth = 10
        n_estimators = 100
        classifier = RandomForestClassifier(n_jobs=-1, max_depth=max_depth, n_estimators=n_estimators)
        # Train the classifier
        classifier.fit(X_train, y_train)

        feature_importance_df = self._calculate_feature_importance(classifier, X_train)

        result_tuples = []

        for num_of_features in self._num_of_features_to_train:
            top_features_df = feature_importance_df.head(num_of_features)
            feature_names_series = top_features_df['Feature_Name']
            top_features = []
            for top_feature_tuple in feature_names_series.items():
                top_feature = top_feature_tuple[1]
                top_features.append(top_feature)

            feature_names = feature_names_series.values

            feature_names_str = "||".join(feature_names)
            column_names = ",".join(top_features)

            # clf = RandomForestClassifier(n_jobs=-1)
            # clf = classifier

            X_training_by_top_features = X_train[top_features]
            X_test_by_top_features = X_test[top_features]
            # Train the classifier

            clf = RandomForestClassifier(n_jobs=-1, max_depth=max_depth, n_estimators=n_estimators)
            clf.fit(X_training_by_top_features, y_train)

            performance_results = self._calculate_performance_for_training_and_test(clf, X_training_by_top_features,
                                                                                    X_test_by_top_features, y_train,
                                                                                    y_test)

            total_results = ("RandomForest", num_of_features, training_size, training_true_num, training_false_num,
                             test_size, test_true_num, test_false_num,) + performance_results + (feature_names_str,)

            result_tuples.append(total_results)
            with open(os.path.join(self._full_path_model_directory,
                                   "RandomForest_by_date_{}_percent_{}_features.model".format(self._training_percent,
                                                                                              num_of_features)),
                      'wb') as model_file:
                pickle.dump(clf, model_file)

            performance_df = pd.DataFrame(result_tuples, columns=['Classifier',
                                                                  'Num_of_Features',
                                                                  'Training_size',
                                                                  '#True(Training)',
                                                                  '#False(Training)',
                                                                  'Test_size',
                                                                  '#True(Test)',
                                                                  '#False(Test)',
                                                                  'AUC Training', 'Accuracy Training',
                                                                  'F1 Training', 'Precision Training',
                                                                  'Recall Training',
                                                                  'AUC Test',
                                                                  'Accuracy Test', 'F1 Test',
                                                                  'Precision Test', 'Recall Test',
                                                                  'Top Features'])

            performance_df.to_csv(
                self._full_path_model_directory + "RandomForest_Results_on_Training_and_Validation_sets_Random_Validation_set_training_{}_percent.csv".format(
                    self._training_percent))

        print("Done")

    def _calculate_performance_for_training_and_test(self, classifier, X_train, X_test, y_train, y_test):
        auc_training = roc_auc_score(y_train, classifier.predict_proba(X_train)[:, 1])
        auc_test = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])

        # fpr, tpr, thresholds = roc_curve(y_train, classifier.predict_proba(X_train)[:, 1])

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
