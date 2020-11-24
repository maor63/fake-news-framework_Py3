# Created by aviade
# Time: 13/04/2017
import copy
from commons.consts import PerformanceMeasures
import itertools
import pandas as pd

class ResultsContainer:
    def __init__(self, path, results_table_file_name, column_names_for_results_table,  *fields):
        self._performance_measures = [PerformanceMeasures.AUC,
                                      PerformanceMeasures.ACCURACY,
                                      PerformanceMeasures.PRECISION,
                                      PerformanceMeasures.RECALL,
                                      PerformanceMeasures.CONFUSION_MATRIX,
                                      PerformanceMeasures.SELECTED_FEATURES,
                                      PerformanceMeasures.CORRECTLY_CLASSIFIED,
                                      PerformanceMeasures.INCORRECTLY_CLASSIFIED
                                      ]

        self._path = path
        self._results_table_file_name = results_table_file_name
        self._column_names_for_results_table = column_names_for_results_table
        self._fields = fields[0]
        self._results_dict = self._create_results_dictionary(fields)

    def _create_results_dictionary(self, fields):

        fields = fields[0][::-1]
        dict = {}

        for performance_measure in self._performance_measures:
            dict[performance_measure] = 0


        for elements in fields:
            temp_dict = copy.deepcopy(dict)
            dict = {}
            for element in elements:
                temprory_dict = copy.deepcopy(temp_dict)
                dict[element] = temprory_dict
        return dict

    def get_results(self):
        return self._results_dict

    def set_result(self, value, key, *elements):
        current_classifier = self._results_dict
        for element in elements:
            current_classifier = current_classifier[element]

        if key is not PerformanceMeasures.SELECTED_FEATURES:
            current_classifier[key] += value
        else:
            current_classifier[key] = value

    def calculate_average_performances(self, num_iterations):
        combinations = list(itertools.product(*self._fields))
        for combination in combinations:
            current_classifier = self._get_to_performance_dict_by_combination(combination)
            # current_classifier = self._results_dict
            # for element in combination:
            #     current_classifier = current_classifier[element]
            for performance_measure in self._performance_measures:
                if not (performance_measure == PerformanceMeasures.SELECTED_FEATURES or
                                performance_measure == PerformanceMeasures.CONFUSION_MATRIX or
                                performance_measure == PerformanceMeasures.CORRECTLY_CLASSIFIED or
                                performance_measure == PerformanceMeasures.INCORRECTLY_CLASSIFIED):
                    current_classifier[performance_measure] /= num_iterations


    def _get_to_performance_dict_by_combination(self, combination):
        current_classifier = self._results_dict
        for element in combination:
            current_classifier = current_classifier[element]
        return current_classifier

    def write_results_as_table(self):
        rows = []

        combinations = list(itertools.product(*self._fields))
        for combination in combinations:
            current_classifier = self._get_to_performance_dict_by_combination(combination)
            row = self._create_row_result(combination, current_classifier)
            rows.append(row)

        dataframe = pd.DataFrame(rows, columns=self._column_names_for_results_table)

        path = self._path + self._results_table_file_name
        dataframe.to_csv(path, index=False)


    def _create_row_result(self, combination, performance_measure_dict):
        row = []
        num_of_correctly_instances = performance_measure_dict[PerformanceMeasures.CORRECTLY_CLASSIFIED]
        num_of_incorrectly_instances = performance_measure_dict[PerformanceMeasures.INCORRECTLY_CLASSIFIED]
        total_instances = num_of_correctly_instances + num_of_incorrectly_instances

        auc = performance_measure_dict[PerformanceMeasures.AUC]
        accuracy = performance_measure_dict[PerformanceMeasures.ACCURACY]
        precision = performance_measure_dict[PerformanceMeasures.PRECISION]
        recall = performance_measure_dict[PerformanceMeasures.RECALL]

        for element in combination:
            row.append(element)

        row.append(num_of_correctly_instances)
        row.append(num_of_incorrectly_instances)
        row.append(total_instances)
        row.append(auc)
        row.append(accuracy)
        row.append(precision)
        row.append(recall)

        return row

    def calculate_correctly_and_not_correctly_instances(self, confusion_matrix_result):
        num_of_correct_instances = 0
        num_of_incorrect_instances = 0
        dimension = confusion_matrix_result.shape[0]
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    num_of_correct_instances += confusion_matrix_result[i][j]
                else:
                    num_of_incorrect_instances += confusion_matrix_result[i][j]

        return num_of_correct_instances, num_of_incorrect_instances

    # def _create_row_result(self, classifier_type_name, num_of_features):
    #     row = []
    #     num_of_correctly_instances = \
    #     self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][PerformanceMeasures.CORRECTLY_CLASSIFIED]
    #     num_of_incorrectly_instances = \
    #     self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][PerformanceMeasures.INCORRECTLY_CLASSIFIED]
    #     total_instances = num_of_correctly_instances + num_of_incorrectly_instances
    #
    #     auc = self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][PerformanceMeasures.AUC]
    #     accuracy = self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][PerformanceMeasures.ACCURACY]
    #     precision = self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][PerformanceMeasures.PRECISION]
    #     recall = self._target_class_classifier_dictionary[self._targeted_class_name][classifier_type_name][num_of_features][PerformanceMeasures.RECALL]
    #
    #     row.append(self._targeted_class_name)
    #     row.append(classifier_type_name)
    #     row.append(num_of_features)
    #     row.append(num_of_correctly_instances)
    #     row.append(num_of_incorrectly_instances)
    #     row.append(total_instances)
    #     row.append(auc)
    #     row.append(accuracy)
    #     row.append(precision)
    #     row.append(recall)
    #
    #     return row


    def find_max_average_auc_classifier(self):
        max_auc = -1
        selected_combination = None
        best_features = None

        combinations = list(itertools.product(*self._fields))
        for combination in combinations:
            current_classifier = self._get_to_performance_dict_by_combination(combination)
            current_auc = current_classifier[PerformanceMeasures.AUC]
            if current_auc > max_auc:
                max_auc = current_auc
                selected_combination = combination
                best_features = current_classifier[PerformanceMeasures.SELECTED_FEATURES]
        return selected_combination, best_features
