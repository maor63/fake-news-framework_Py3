import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

class Dataframe_Manipulator():
    def __init__(self, replace_missing_values, indentifier_field_name, targeted_class_field_name, feature_names_to_remove,
                 selected_features):
        self._replace_missing_values = replace_missing_values
        self._indentifier_field_name = indentifier_field_name
        self._targeted_class_field_name = targeted_class_field_name
        self._feature_names_to_remove = feature_names_to_remove
        self._selected_features = selected_features

    def prepare_dataframe_for_learning(self, dataframe):
        dataframe.reset_index(drop=True, inplace=True)

        # Replace None in 0 for later calculation
        if self._replace_missing_values == 'zero':
            dataframe = dataframe.fillna(0)
        elif self._replace_missing_values == 'mean':
            dataframe = dataframe.fillna(dataframe.mean())

        #dataframe = replace_nominal_class_to_numeric(dataframe, self._optional_classes)

        indentifier_series = dataframe.pop(self._indentifier_field_name)
        targeted_class_series = dataframe.pop(self._targeted_class_field_name)

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


    def reduce_dimensions_by_num_of_features(self, labeled_author_features_dataframe, targeted_class_series,
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

    def _get_k_best_feature_names(self, k_best_classifier, original_dataframe):
        mask = k_best_classifier.get_support()
        best_feature_names = []
        column_names = list(original_dataframe.columns.values)
        for boolean_value, feature_name in zip(mask, column_names):
            if boolean_value == True:
                best_feature_names.append(feature_name)
        return best_feature_names

    def calculate_features_to_remove(self, best_feature_names, original_feature_names):
        best_combination_set = set(best_feature_names)
        original_feature_names_set = set(original_feature_names)
        features_to_remove_set = original_feature_names_set - best_combination_set
        features_to_remove = list(features_to_remove_set)
        return features_to_remove

    def remove_features(self, features_to_remove, dataframe):
        '''
        This function is responsible to remove features.
        :param dataframe:
        :return:dataframe without removed columns
        '''
        #print("Start remove_unnecessary_features")
        # Remove unnecessary features
        dataframe_columns = dataframe.columns
        for unnecessary_feature in features_to_remove:
            if unnecessary_feature in dataframe_columns:
                dataframe.pop(unnecessary_feature)

        return dataframe


