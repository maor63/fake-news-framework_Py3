# Written by Lior Bass 1/4/2018

import csv
import logging
import pickle
import joblib
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from commons.consts import PerformanceMeasures
from experimental_environment.refactored_experimental_enviorment.classifier_runner import Classifier_Runner
from experimental_environment.refactored_experimental_enviorment.data_handler import Data_Handler
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
##classifiers import-- DO NOT DELETE!!!!!!!!
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, BaggingClassifier


##END
class Classifier_Trainer(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._target_field = self._config_parser.eval(self.__class__.__name__, "target_field")
        self._select_features = self._config_parser.eval(self.__class__.__name__, "select_features")
        self._remove_features = self._config_parser.eval(self.__class__.__name__, "remove_features")
        self._k = self._config_parser.eval(self.__class__.__name__, "k")
        self._num_of_features_to_select = self._config_parser.eval(self.__class__.__name__, "num_of_features_to_select")
        self._classifiers_with_parameters_dict = self._config_parser.eval(self.__class__.__name__,
                                                                          "classifiers_with_parameters_dict")
        self._compare_matics_by_order = self._config_parser.eval(self.__class__.__name__, "compare_matrics_by_order")
        self._label_text_to_value = self._config_parser.eval(self.__class__.__name__, "label_text_to_value")
        self._results_file_path = "data/output/expermintal_environment/classification_results_refactored.csv"
        self._feature_importance_file ='data\\output\\expermintal_environment\\feature_importance.csv'
        self._saved_classifier_path = self._config_parser.eval(self.__class__.__name__, "saved_classifier_path")
        self._results_dictionary = {}
        self._classifiers_by_name = {}
        self._data_handler = Data_Handler(db, targeted_class_name=self._target_field)

    def execute(self, window_start=None):
        logging.info("started training classifiers")
        authors_features_dataframe, authors_labels = self._data_handler.get_labeled_authors_feature_dataframe_for_classification(self._remove_features, self._select_features, self._label_text_to_value)
        # self._calculate_feature_variance(authors_features_dataframe) # this is for test reasons
        self.calculate_feature_importance(authors_features_dataframe, authors_labels)
        print("==============================")
        for num_of_features in self._num_of_features_to_select:
            reduced_authors_features_dataframe, selected_feature_names = self._find_k_best_and_reduce_dimensions(
                num_of_features, authors_features_dataframe, authors_labels)
            for classifier_dictionary in self._classifiers_with_parameters_dict:
                for number_of_fragments in self._k:
                    classifier_name = classifier_dictionary["name"]
                    msg = "\r Classifier name: {0}, Number of features to select {1}, k = {2}".format(classifier_name,
                                                                                                      num_of_features,
                                                                                                      number_of_fragments)
                    print(msg, end="")
                    current_experiment_result_sum = Result_Container()
                    for fragment_index in range(number_of_fragments):
                        current_experiment_result_sum = self.run_experiment_on_fragment(authors_labels,
                                                                                        classifier_dictionary,
                                                                                        current_experiment_result_sum,
                                                                                        reduced_authors_features_dataframe,
                                                                                        fragment_index,
                                                                                        number_of_fragments)
                    classifier_result = Classifier_Result(number_of_fragments, num_of_features,
                                                          len(authors_features_dataframe.columns),
                                                          classifier_dictionary["name"],
                                                          classifier_dictionary["params"], selected_feature_names)
                    self.calculate_average_result_of_classification_on_fragments(classifier_dictionary,
                                                                                 current_experiment_result_sum,
                                                                                 number_of_fragments, classifier_result,
                                                                                 num_of_features)
        self._save_results_to_csv(self._results_file_path)
        self.summarize_and_score_best_classifier(authors_features_dataframe, authors_labels)
        exit(0)

    def summarize_and_score_best_classifier(self, authors_features_dataframe, authors_labels):
        best_classifier_name, best_classifier_results = self.find_best_classifier()
        print(
        "==============================================\n==============================================\n==============================================")
        print("best_classifier= " + best_classifier_results.classifier_name + " \nresults: " +
              best_classifier_results.to_string())
        reduced_authors_features_dataframe, selected_feature_names = self._find_k_best_and_reduce_dimensions(
            best_classifier_results.num_of_feature_selected, authors_features_dataframe, authors_labels)
        classifier = self._classifiers_by_name[best_classifier_name].fit(reduced_authors_features_dataframe,
                                                                         authors_labels)
        logging.info("saving best classifier pickle")
        self._save_to_disk(classifier, best_classifier_results.classifier_name + '.pkl')
        selected_features_names = best_classifier_results.selected_features_names
        self._save_to_disk(selected_features_names, 'selected_features_names.pkl')

    def calculate_average_result_of_classification_on_fragments(self, classifier_dictionary,
                                                                current_experiment_result_sum, k, classifier_results,
                                                                num_of_features):
        experiment_name = self.get_experminet_name(k, classifier_dictionary["name"],
                                                   classifier_dictionary["params"], num_of_features)
        current_experiment_avg_result = current_experiment_result_sum.divide_by_scalar(k)
        classifier_results.set_results(current_experiment_avg_result)
        logging.info("current result:" + classifier_results.to_string())
        self._classifiers_by_name[experiment_name] = self.get_classifier_instance(classifier_dictionary)
        self._results_dictionary[experiment_name] = classifier_results

    def run_experiment_on_fragment(self, authors_labels, classifier_dictionary, current_experiment_result_sum,
                                   data_frame, i, k):
        test_set, train_set, test_labels, train_labels = self._data_handler.get_the_k_fragment_from_dataset(data_frame,
                                                                                                            authors_labels,
                                                                                                            i, k)
        fitted_classifier = self.run_classifier(classifier_dictionary, train_set, train_labels)
        current_results = self.evaluate_classifier(fitted_classifier, test_set, test_labels)
        current_experiment_result_sum = current_experiment_result_sum.add(current_results)
        return current_experiment_result_sum

    def get_experminet_name(self, k, classifier, parameters, num_of_features, **kwargs):
        experiment_str = "K: " + str(k) + "_classifierName:" + str(classifier) + "_parameters:" + str(
            parameters) + "_numOfFeatures:" + str(num_of_features)
        for arg in kwargs:
            experiment_str += " " + str(arg)
        return experiment_str

    def run_classifier(self, classifier_dict, X, Y):
        classifier = self.get_classifier_instance(classifier_dict)
        # logging.info("started fitting classifier - "+classifier_name)
        classifier.fit(X, Y)
        # logging.info("finished fitting classifier - " + classifier_name)
        return classifier

    def get_classifier_instance(self, classifier_dict):
        classifier_name = classifier_dict["name"]
        params = classifier_dict["params"]
        classifier = eval(classifier_name + "(" + params + ")")
        return classifier

    def evaluate_classifier(self, classifier, test_features, test_labels):
        predicted = classifier.predict(test_features)
        actual = test_labels

        performance_report = precision_recall_fscore_support(actual, predicted, average='weighted')
        precision = performance_report[0]
        recall = performance_report[1]
        f1 = performance_report[2]
        accuracy = accuracy_score(actual, predicted)
        try:
            # num_of_classes= len(set(actual))
            # # num_of_classes = predicted.unique()
            # if num_of_classes>2:
            #     auc = self._calculate_weighted_auc(PerformanceMeasures.AUC, actual,predicted)
            # elif num_of_classes==2:
                auc = roc_auc_score(actual, predicted)
        except Exception as e:
            auc = -1
        results = Result_Container(precision, recall, accuracy, auc, f1)
        return results

    def find_best_classifier(self):
        best_classifier_name_and_scores = ('empty_classifier', Classifier_Result())
        for classifier_name in list(self._results_dictionary.keys()):
            classifier_results = self._results_dictionary[classifier_name]
            current_classifier_result_tupple = (classifier_name, classifier_results)
            best_classifier_name_and_scores = self._compare_results(best_classifier_name_and_scores,
                                                                    current_classifier_result_tupple)
        best_classifier = best_classifier_name_and_scores[0]
        classifier_results = best_classifier_name_and_scores[1]
        return best_classifier, classifier_results

    def _find_k_best_and_reduce_dimensions(self, num_of_features, labeled_author_features_dataframe,
                                           targeted_class_series):
        if num_of_features == 'all':
            return labeled_author_features_dataframe, labeled_author_features_dataframe.columns
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

    def _compare_results(self, result1, result2):
        result_1_first_val, result_1_second_val, result_2_first_val, result_2_second_val = self.get_comparable_results_fields(
            result1, result2)
        if result_1_first_val == result_2_first_val:
            if result_1_second_val > result_2_second_val:
                return result1
            else:
                return result2
        elif result_1_first_val > result_2_first_val:
            return result1
        return result2

    def _get_k_best_feature_names(self, k_best_classifier, original_dataframe):
        mask = k_best_classifier.get_support()
        best_feature_names = []
        column_names = list(original_dataframe.columns.values)
        for boolean_value, feature_name in zip(mask, column_names):
            if boolean_value == True:
                best_feature_names.append(feature_name)
        return best_feature_names

    def get_comparable_results_fields(self, classifier_1_result, classifier_2_result):
        first_param = self._compare_matics_by_order[0]
        second_param = self._compare_matics_by_order[1]
        result1 = classifier_1_result[1].result_container
        result2 = classifier_2_result[1].result_container
        result_1_first_val = getattr(result1, first_param)
        result_2_first_val = getattr(result2, first_param)
        result_1_second_val = getattr(result1, second_param)
        result_2_second_val = getattr(result2, second_param)
        return result_1_first_val, result_1_second_val, result_2_first_val, result_2_second_val

    def _save_results_to_csv(self, _results_file_path):
        with open(self._results_file_path, 'w') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow(Classifier_Result.columns)
            for cdr in list(self._results_dictionary.values()):
                row = cdr.to_list()
                wr.writerow(row)

    def _save_to_disk(self, classifier, classifier_name):
        file_path = self._saved_classifier_path + "/" + classifier_name
        with open(file_path, 'wb') as fid:
            joblib.dump(classifier, fid)

    def _calculate_feature_variance(self, labeled_author_features_dataframe):
        variance_list = []
        for column in labeled_author_features_dataframe.columns:
            value = labeled_author_features_dataframe[column].var()
            variance_list.append(value)
        print(variance_list)
        return variance_list

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

    def calculate_feature_importance(self, X, y):
        # X, y = make_classification(n_samples=1000,
        #                            n_features=10,
        #                            n_informative=3,
        #                            n_redundant=0,
        #                            n_repeated=0,
        #                            n_classes=2,
        #                            random_state=0,
        #                            shuffle=False)

        # Build a forest and compute the feature importances
        column_headers = list(X.columns.values)
        forest = ExtraTreesClassifier(n_estimators=len(column_headers),
                                      random_state=0, criterion='entropy')
        forest.fit(X, y)
        # column_headers = forest.estimator_params

        importances = forest.feature_importances_
        std = np.std([tree.tree_.compute_feature_importances(normalize=False) for tree in forest.estimators_],
                     axis=0)
        # indices = np.argsort(std)[::-1]

        # Print the feature ranking
        csv_file= open(self._feature_importance_file, 'wb')
        wr = csv.writer(csv_file, delimiter=',')
        wr.writerow(['feature', 'score'])

        print("Feature ranking:")

        for f in range(X.shape[1]):
            row = [column_headers[f]]
            # row.append(importances[indices[f]])
            row.append(std[f])
            wr.writerow(row)
            # print(str(f + 1)+" feature " +str(column_headers[f])+"   " +str(importances[indices[f]]))
            print(str(f + 1) + " feature " + str(column_headers[f]) + "   " + str(std[f]))
        csv_file.close()
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(list(range(X.shape[1])), std,
                color="r", yerr=std, align="center")
        plt.xticks(list(range(X.shape[1])), column_headers)
        plt.xlim([-1, X.shape[1]])
        #plt.show()

class Result_Container():
    columns = ['precision', 'recall', 'accuracy', 'auc', 'f1']
    precision = 0.0
    recall = 0.0
    accuracy = 0.0
    auc = 0.0
    f1 = 0.0

    def __init__(self, precision=0.0, recall=0.0, accuracy=0.0, auc=0.0, f1=0.0):
        self.precision = round(precision,2)
        self.recall = round(recall,2)
        self.accuracy = round(accuracy,2)
        self.auc = round(auc,2)
        self.f1 = round(f1,2)

    def add(self, result_container):
        self.precision += result_container.precision
        self.recall += result_container.recall
        self.accuracy += result_container.accuracy
        self.auc += result_container.auc
        self.f1 += result_container.f1
        return self

    def divide_by_scalar(self, n):
        self.precision = round(self.precision / n,2)
        self.recall = round(self.recall / n,2)
        self.accuracy = round(self.accuracy / n,2)
        self.auc = round(self.auc / n,2)
        self.f1 = round(self.f1 / n,2)
        return self

    def to_list(self):
        ans = [self.precision, self.recall, self.accuracy, self.auc, self.f1]
        return ans

    def to_string(self):
        return "precision: " + str(self.precision) + " recall: " + str(self.recall) + " accuracy: " + str(
            self.accuracy) + " auc: " + str(self.auc) + " f1: " + str(self.f1)

    def __repr__(self):
        return self.to_string()


class Classifier_Result():
    columns = ["classifier name", "classifier parameters", "k", "number of selected features",
               "number of available features", "selected features"] + Result_Container.columns
    result_container = Result_Container()
    k = 0
    num_of_feature_selected = 0
    num_of_available_feature = 0
    classifier_name = ""
    classifier_parameters = ""
    selected_features_names = ""

    def __init__(self, k=0, num_of_feature_selected=0, number_of_available_features=0, classifier_name="",
                 classifier_parameters="", selected_features_names=""):
        self.k = k
        self.result_container = Result_Container()
        self.num_of_feature_selected = num_of_feature_selected
        self.num_of_available_feature = number_of_available_features
        self.classifier_name = classifier_name
        self.classifier_parameters = classifier_parameters
        self.selected_features_names = selected_features_names

    def set_results(self, result_container):
        self.result_container = result_container

    def to_list(self):
        ans = [self.classifier_name, self.classifier_parameters, self.k, self.num_of_feature_selected,
               self.num_of_available_feature, self.selected_features_names]
        ans = ans + self.result_container.to_list()
        return ans

    def to_string(self):
        return "classifier name: " + str(self.classifier_name) + \
               " \nclassifier_parameters: " + str(self.classifier_parameters) + \
               "\n num of features selected: " + str(
            self.num_of_feature_selected) + " num of available feature: " + str(
            self.num_of_available_feature) + " result_container: " + str(self.result_container.to_string())

    def __repr__(self):
        return self.to_string()

