
import pandas as pd
import networkx as nx
import urllib.request, urllib.error, urllib.parse
from commons.method_executor import Method_Executor
from sklearn.metrics import precision_score, accuracy_score, recall_score, precision_recall_fscore_support, f1_score
import editdistance
import jellyfish


__author__ = "Aviad Elyashar"


class NameAnalyzer(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_directory_path = self._config_parser.eval(self.__class__.__name__, "input_directory_path")
        self._output_directory_path = self._config_parser.eval(self.__class__.__name__, "output_directory_path")
        self._results_file_name = self._config_parser.eval(self.__class__.__name__, "results_file_name")
        self._target_file_name = self._config_parser.eval(self.__class__.__name__, "target_file_name")
        self._name_synonym_ground_truth_file = self._config_parser.eval(self.__class__.__name__, "name_synonym_ground_truth_file")
        self._targeted_sound_measure = self._config_parser.eval(self.__class__.__name__, "targeted_sound_measure")
        self._selected_string_similarity_function = self._config_parser.eval(self.__class__.__name__, "selected_string_similarity_function")

    def create_synonyms_for_graph_first_names_by_soundex(self):
        dfs = []

        first_names_graph_synonyms_df = pd.read_csv(self._input_directory_path + "first_names_graph_synonyms_ground_truth_3.csv")
        name_measures_df = pd.read_csv(self._input_directory_path + "sound_performance_on_all_names.csv")

        first_names_series = first_names_graph_synonyms_df["Name"]
        first_names_series = first_names_series.drop_duplicates()
        first_names = first_names_series.tolist()
        for i, first_name in enumerate(first_names):
            print("\rFirst Name: {0} {1}/{2}".format(first_name, i, len(first_names)), end='')

            first_name_with_measures_df = name_measures_df[name_measures_df["Name"] == first_name]

            if first_name_with_measures_df.empty:
                continue

            first_name_soundex = first_name_with_measures_df["Name_Soundex"]
            soundex = first_name_soundex.values[0]
            first_name_with_measures_df = name_measures_df[name_measures_df["Name_Soundex"] == soundex]
            first_name_with_measures_df["Source_Name"] = first_name

            first_name_synonyms_by_soundex_df = first_name_with_measures_df[['Source_Name', 'Name', 'Name_Soundex']]
            dfs.append(first_name_synonyms_by_soundex_df)

        results_df = pd.concat(dfs)
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def fill_results_by_ground_truth(self):
        results = []
        # synonyms by soundex
        first_names_graph_synonyms_df = pd.read_csv(
            self._input_directory_path + self._target_file_name)

        # # synonyms by our method
        # first_names_graph_synonyms_df = pd.read_csv(
        #          self._input_directory_path + "first_names_child_ancestor_count_greater_than_1_aggregated.csv")

        graph_first_name_synonym_ground_truth_df = pd.read_csv(
            self._input_directory_path + self._name_synonym_ground_truth_file)

        for index, row in first_names_graph_synonyms_df.iterrows():
            source_name = row["Source_Name"]
            synonym = row["Name"]
            num_of_rows = first_names_graph_synonyms_df.shape[0]
            print("\rFirst Name: {0} {1}/{2}".format(source_name, index, num_of_rows), end='')

            result_df = graph_first_name_synonym_ground_truth_df[(graph_first_name_synonym_ground_truth_df["Name"] == source_name) &
                                                     (graph_first_name_synonym_ground_truth_df["Synonym"] == synonym)]

            if result_df.empty:
                result = (source_name, synonym, 0)
            else:
                result = (source_name, synonym, 1)

            results.append(result)

        results_df = pd.DataFrame(results, columns=['Source_Name', 'Synonym', 'Is_Original_Synonym'])
        results_df.to_csv(self._output_directory_path + self._results_file_name)


    def fill_results_for_rival_by_ground_truth(self):
        results = []
        # synonyms by soundex
        first_names_graph_synonyms_df = pd.read_csv(
            self._input_directory_path + self._target_file_name)

        # # synonyms by our method
        # first_names_graph_synonyms_df = pd.read_csv(
        #          self._input_directory_path + "first_names_child_ancestor_count_greater_than_1_aggregated.csv")

        graph_first_name_synonym_ground_truth_df = pd.read_csv(
            self._input_directory_path + self._name_synonym_ground_truth_file)

        for index, row in first_names_graph_synonyms_df.iterrows():
            source_name = row["Source_Name"]
            synonym = row["Name"]
            num_of_rows = first_names_graph_synonyms_df.shape[0]
            print("\rFirst Name: {0} {1}/{2}".format(source_name, index, num_of_rows), end='')

            result_df = graph_first_name_synonym_ground_truth_df[(graph_first_name_synonym_ground_truth_df["Name"] == source_name) &
                                                     (graph_first_name_synonym_ground_truth_df["Synonym"] == synonym)]

            if result_df.empty:
                result = (source_name, synonym, 0)
            else:
                result = (source_name, synonym, 1)

            results.append(result)

        results_df = pd.DataFrame(results, columns=['Source_Name', 'Synonym', 'Is_Original_Synonym'])
        results_df.to_csv(self._output_directory_path + self._results_file_name)


    def compare_suggestions_with_ground_truth(self):
        ground_truth_df = pd.read_csv(self._input_directory_path + self._name_synonym_ground_truth_file)
        suggestions_df = pd.read_csv(self._input_directory_path + self._target_file_name)

        suggestions_with_ground_truth_df = self._compare_suggestions_with_ground_truth_by_provided_dfs(suggestions_df, ground_truth_df)
        print("Done compare_suggestions_with_ground_truth!!!!!")
        return suggestions_with_ground_truth_df

    def _compare_suggestions_with_ground_truth_by_provided_dfs(self, suggestions_df, ground_truth_df):
        suggestions_df['Is_Original_Synonym'] = suggestions_df.apply(
            lambda x: compare_suggestion(x["Original"], x["Candidate"], ground_truth_df),
            axis=1)
        # suggestions_df.to_csv(self._output_directory_path + "Suggesting_names_by_{0}_with_ground_truth.csv".format(
        #     self._targeted_sound_measure), index=False)
        suggestions_df.to_csv(self._output_directory_path + self._results_file_name, index=False)
        return suggestions_df


    def sort_file_by_names(self):
        # synonyms by our method
        name_synonym_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        name_synonym_df = name_synonym_df.sort_values(by=['Source_Name'])

        name_synonym_df.to_csv(self._output_directory_path + self._results_file_name, index=False)


    ###
    # 1. Move on the names found by our method (Edit distance = 1)
    # 2. For each name bring the soundex and names that are similar.
    # 3. Find if it is really synonyms
    ##
    def create_rival_sound_measure_results_by_graph_first_name_synonym_edit_distance(self):
        dfs = []
        # synonyms by our method
        graph_first_name_edit_distance_1_df = pd.read_csv(
                 self._input_directory_path + "first_names_child_ancestor_count_greater_than_1_aggregated.csv")

        child_first_name_series = graph_first_name_edit_distance_1_df["Source_Name"]
        child_first_names = child_first_name_series.tolist()


        names = list(set(child_first_names))
        sorted_names = [name for name in names if len(name) > 2]
        sorted_names = sorted(sorted_names)
        sorted_names_df = pd.DataFrame(sorted_names, columns=['Name:'])
        sorted_names_df.to_csv(self._output_directory_path + "sorted_names_first_names_child_ancestor_count_greater_than_1_aggregated.csv", index=False)


        name_sound_performance_df = pd.read_csv(
            #self._input_directory_path + "sound_performance_on_all_names2.csv")
            self._input_directory_path + "sound_performance_including_nysiis_on_all_names.csv")

        for i, name in enumerate(sorted_names):
            print("\rFirst Name: {0} {1}/{2}".format(name, i, len(sorted_names)), end='')
            selected_name_sound_performance_df = name_sound_performance_df[name_sound_performance_df["Name"] == name]

            if selected_name_sound_performance_df.empty:
                continue

            #selected_name_soundex = selected_name_sound_performance_df["Soundex"]
            selected_name_sound_measure = selected_name_sound_performance_df[self._targeted_sound_measure]
            sound_measure_result = selected_name_sound_measure.values[0]

            name_with_the_same_sound_measure_df = name_sound_performance_df[(name_sound_performance_df[self._targeted_sound_measure] == sound_measure_result) &
                                                                       (name_sound_performance_df["Name"] != name)]

            name_with_the_same_sound_measure_df["Source_Name"] = name

            first_name_synonyms_by_soundex_df = name_with_the_same_sound_measure_df[['Source_Name', 'Name', self._targeted_sound_measure]]


            dfs.append(first_name_synonyms_by_soundex_df)

        results_df = pd.concat(dfs)
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def create_rival_sound_measure_suggestions(self):
        ground_truth_df = pd.read_csv(self._input_directory_path + self._name_synonym_ground_truth_file)
        original_names_series = ground_truth_df["Name"]
        original_names = original_names_series.tolist()
        original_names = list(set(original_names))
        original_names = sorted(original_names)

        name_sound_performance_df = pd.read_csv(
            self._input_directory_path + "sound_performance_including_nysiis_on_all_names.csv")

        dfs = []
        for i, original_name in enumerate(original_names):
            print("\rSuggesting candidates for: {0} {1}/{2}".format(original_name, i, len(original_names)), end='')
            selected_name_sound_performance_df = name_sound_performance_df[name_sound_performance_df["Name"] == original_name]

            if selected_name_sound_performance_df.empty:
                continue


            #selected_name_soundex = selected_name_sound_performance_df["Soundex"]
            selected_name_sound_measure = selected_name_sound_performance_df[self._targeted_sound_measure]
            sound_measure_result = selected_name_sound_measure.values[0]

            name_with_the_same_sound_measure_df = name_sound_performance_df[(name_sound_performance_df[self._targeted_sound_measure] == sound_measure_result) &
                                                                       (name_sound_performance_df["Name"] != original_name)]

            if name_with_the_same_sound_measure_df.empty:
                continue

            name_with_the_same_sound_measure_df['Edit_Distance'] = name_with_the_same_sound_measure_df.apply(
                lambda x: calculate_edit_distance(original_name, x["Name"]),
                axis=1)

            name_with_the_same_sound_measure_df = name_with_the_same_sound_measure_df.sort_values(by='Edit_Distance')

            # Take only first 10 for each name
            #name_with_the_same_sound_measure_df = name_with_the_same_sound_measure_df.head(10)


            name_with_the_same_sound_measure_df["Source_Name"] = original_name

            first_name_synonyms_by_soundex_df = name_with_the_same_sound_measure_df[['Source_Name', 'Name', self._targeted_sound_measure, 'Edit_Distance']]
            first_name_synonyms_by_soundex_df = first_name_synonyms_by_soundex_df.rename(columns={'Source_Name': 'Original', 'Name': 'Candidate'})



            dfs.append(first_name_synonyms_by_soundex_df)

        results_df = pd.concat(dfs)
        #results_df = results_df.sort_values(by='Original')
        results_df.to_csv(self._output_directory_path + "Suggesting_names_by_{0}.csv".format(self._targeted_sound_measure), index=False)
        #results_df.to_csv(self._output_directory_path + "Suggesting_names_by_{0}_and_edit_distance.csv".format(self._targeted_sound_measure), index=False)

        print("Done create_rival_sound_measure_suggestions!!!!!")
        return results_df, ground_truth_df

    def calculate_performance_per_name_for_name_synonym(self):
        # synonyms by our method
        targeted_results_df = pd.read_csv(self._input_directory_path + self._target_file_name)

        ground_truth_df = pd.read_csv(self._input_directory_path + self._name_synonym_ground_truth_file)

        source_names_series = targeted_results_df["Source_Name"]
        source_names = source_names_series.tolist()
        source_names = list(set(source_names))
        source_names = sorted(source_names)

        final_results = []
        for i, source_name in enumerate(source_names):
            print("\rFirst Name: {0} {1}/{2}".format(source_name, i, len(source_names)), end='')

            source_name_results_df = targeted_results_df[targeted_results_df["Source_Name"] == source_name]
            predictions = source_name_results_df["Is_Original_Synonym"]

            num_of_rows = source_name_results_df.shape[0]
            actual = [1] * num_of_rows

            accuracy = accuracy_score(actual, predictions)
            f1 = f1_score(actual, predictions)
            precison = precision_score(actual, predictions, average='micro')

            source_name_ground_truth_df = ground_truth_df[ground_truth_df["Name"] == source_name]
            source_name_num_of_relevant_synonyms = source_name_ground_truth_df.shape[0]

            num_of_relevant_retrieved = predictions.sum()
            num_of_retrieved = predictions.count()

            recall_related_to_ground_truth = -1
            if source_name_num_of_relevant_synonyms > 0:
                recall_related_to_ground_truth = num_of_relevant_retrieved / float(source_name_num_of_relevant_synonyms)
            precision_related_to_ground_truth = num_of_relevant_retrieved / float(num_of_retrieved)

            recall = recall_score(actual, predictions)

            result_tuple = (source_name, num_of_relevant_retrieved, num_of_retrieved,
                            source_name_num_of_relevant_synonyms, accuracy, f1, precison,
                            precision_related_to_ground_truth, recall, recall_related_to_ground_truth)
            final_results.append(result_tuple)

        final_results_df = pd.DataFrame(final_results, columns=['{0}_Source_Name'.format(self._targeted_sound_measure),
                                                                '{0}_Num of Relevant Retrieved'.format(self._targeted_sound_measure),
                                                                '{0}_Num of Retrieved'.format(self._targeted_sound_measure),
                                                                '{0}_Total Num of Relevant in Ground Truth'.format(self._targeted_sound_measure),
                                                                '{0}_Accuracy'.format(self._targeted_sound_measure),
                                                                '{0}_F1'.format(self._targeted_sound_measure),
                                                                '{0}_Precision'.format(self._targeted_sound_measure),
                                                                '{0}_Precision_Calculated_by_Me'.format(self._targeted_sound_measure),
                                                                '{0}_Recall'.format(self._targeted_sound_measure),
                                                                '{0}_Recall_Calculated_by_Me'.format(self._targeted_sound_measure)])
        final_results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!")

    def calculate_performance_for_suggestions(self):
        suggestions_df = pd.read_csv(self._input_directory_path + self._target_file_name)

        ground_truth_df = pd.read_csv(self._input_directory_path + self._name_synonym_ground_truth_file)

        self._calculate_performance(suggestions_df, ground_truth_df)

        print("Done!!!")

    def _calculate_performance(self, suggestions_df, ground_truth_df):
        source_names_series = suggestions_df["Original"]
        # source_names_series = suggestions_df["Source_Name"]
        source_names = source_names_series.tolist()
        source_names = list(set(source_names))
        source_names = sorted(source_names)

        final_results = []
        for i, source_name in enumerate(source_names):
            print("\rFirst Name: {0} {1}/{2}".format(source_name, i, len(source_names)), end='')

            source_name_results_df = suggestions_df[suggestions_df["Original"] == source_name]
            # source_name_results_df = suggestions_df[suggestions_df["Source_Name"] == source_name]
            predictions = source_name_results_df["Is_Original_Synonym"]

            num_of_rows = source_name_results_df.shape[0]
            actual = [1] * num_of_rows

            accuracy = accuracy_score(actual, predictions)
            predictions_10 = predictions[0:10]
            actual_10 = actual[0:10]
            accuracy_10 = accuracy_score(actual_10, predictions_10)

            f1 = f1_score(actual, predictions)
            predictions_10 = predictions[0:10]
            actual_10 = actual[0:10]
            f1_10 = f1_score(actual_10, predictions_10)

            precison = precision_score(actual, predictions, average='micro')

            precison_1, precison_2, precison_3, precison_5, precision_10 = self._calculte_precision_at(actual,
                                                                                                       predictions)

            source_name_ground_truth_df = ground_truth_df[ground_truth_df["Name"] == source_name]
            source_name_num_of_relevant_synonyms = source_name_ground_truth_df.shape[0]

            num_of_relevant_retrieved_at_10 = predictions_10.sum()
            num_of_retrieved_at_10 = predictions_10.count()

            num_of_relevant_retrieved = predictions.sum()
            num_of_retrieved = predictions.count()



            recall_related_to_ground_truth = -1
            if source_name_num_of_relevant_synonyms > 0:
                recall_related_to_ground_truth = num_of_relevant_retrieved / float(source_name_num_of_relevant_synonyms)

            recall_1, recall_2, recall_3, recall_5, recall_10 = self._calculate_recall_at(predictions,
                                                                                          source_name_num_of_relevant_synonyms)

            # precision_related_to_ground_truth = num_of_relevant_retrieved / float(num_of_retrieved)

            # recall = recall_score(actual, predictions)

            result_tuple = (source_name, num_of_relevant_retrieved, num_of_retrieved, num_of_relevant_retrieved_at_10,
                            num_of_retrieved_at_10,
                            source_name_num_of_relevant_synonyms, accuracy, accuracy_10, f1, f1_10,
                            precison_1, precison_2, precison_3, precison_5, precision_10, precison,
                            recall_1, recall_2, recall_3, recall_5, recall_10, recall_related_to_ground_truth)
            final_results.append(result_tuple)

        final_results_df = pd.DataFrame(final_results, columns=['Source_Name',
                                                                'Num of Relevant Retrieved',
                                                                'Num of Retrieved',
                                                                'Num of Relevant Retrieved@10',
                                                                'Num of Retrieved@10',
                                                                'Total Num of Relevant in Ground Truth',
                                                                'Accuracy',
                                                                'Accuracy@10',
                                                                'F1',
                                                                'F1@10',
                                                                'Precision@1',
                                                                'Precision@2',
                                                                'Precision@3',
                                                                'Precision@5',
                                                                'Precision@10',
                                                                'Precision',
                                                                'Recall@1',
                                                                'Recall@2',
                                                                'Recall@3',
                                                                'Recall@5',
                                                                'Recall@10',
                                                                'Recall'])
        #final_results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        # final_results_df.to_csv(self._output_directory_path + "Suggesting_names_by_{0}_with_ground_truth_performance.csv".format(
        #     self._targeted_sound_measure), index=False)

        final_results_df.to_csv(
            self._output_directory_path + self._results_file_name, index=False)


        print("Done!!!")

    def _calculte_precision_at(self, actual, predictions):
        # precision_at_results = ()
        # for num in [1, 2, 3, 5]:
        #     predictions_num = predictions.head(num)
        #     actual_num = actual.head(num)
        #     precison_num = precision_score(actual_num, predictions_num, average='micro')
        #     precision_at_results = precision_at_results + (precison_num,)
        # return precision_at_results

        predictions_1 = predictions[0:1]
        actual_1 = actual[0:1]
        precison_1 = precision_score(actual_1, predictions_1, average='micro')

        predictions_2 = predictions[0:2]
        actual_2 = actual[0:2]
        precison_2 = precision_score(actual_2, predictions_2, average='micro')

        predictions_3 = predictions[0:3]
        actual_3 = actual[0:3]
        precison_3 = precision_score(actual_3, predictions_3, average='micro')

        predictions_5 = predictions[0:5]
        actual_5 = actual[0:5]
        precison_5 = precision_score(actual_5, predictions_5, average='micro')

        predictions_10 = predictions[0:10]
        actual_10 = actual[0:10]
        precison_10 = precision_score(actual_10, predictions_10, average='micro')

        return precison_1, precison_2, precison_3, precison_5, precison_10


    def _calculate_recall_at(self, predictions, source_name_num_of_relevant_synonyms):
        num_of_relevant_retrieved_1 = predictions[0:1].sum()
        recall_1 = num_of_relevant_retrieved_1 / float(source_name_num_of_relevant_synonyms)

        num_of_relevant_retrieved_2 = predictions[0:2].sum()
        recall_2 = num_of_relevant_retrieved_2 / float(source_name_num_of_relevant_synonyms)

        num_of_relevant_retrieved_3 = predictions[0:3].sum()
        recall_3 = num_of_relevant_retrieved_3 / float(source_name_num_of_relevant_synonyms)

        num_of_relevant_retrieved_5 = predictions[0:5].sum()
        recall_5 = num_of_relevant_retrieved_5 / float(source_name_num_of_relevant_synonyms)

        num_of_relevant_retrieved_10 = predictions[0:10].sum()
        recall_10 = num_of_relevant_retrieved_10 / float(source_name_num_of_relevant_synonyms)

        return recall_1, recall_2, recall_3, recall_5, recall_10

        # two files - our method + soundex
    def unite_performance_results_given_two_methods(self):

        graph_edit_distance_results_df = pd.read_csv(self._input_directory_path + "graph_first_names_synonyms_by_edit_distance_1_sorted_by_source_name_results_performance.csv")
        #soundex_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_soundex_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_soundex_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_metaphone_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_united_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_nysiis_results_performance.csv")
        target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_name_suggestions_by_match_rating_codex_with_ground_truth_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_secondary_results_performance.csv")

        # remove names that are not exist in the ground truth
        graph_edit_distance_results_df = graph_edit_distance_results_df[graph_edit_distance_results_df["Edit_Distance_Total Num of Relevant in Ground Truth"] != 0]
        target_sound_measure_results_df = target_sound_measure_results_df[target_sound_measure_results_df["{0}_Total Num of Relevant in Ground Truth".format(self._targeted_sound_measure)] != 0]

        graph_edit_distance_names_series = graph_edit_distance_results_df["Edit_Distance_Source_Name"]
        graph_edit_distance_names = graph_edit_distance_names_series.tolist()

        soundex_names_series = target_sound_measure_results_df["{0}_Source_Name".format(self._targeted_sound_measure)]
        soundex_names = soundex_names_series.tolist()

        names = graph_edit_distance_names + soundex_names
        names = list(set(names))
        names = sorted(names)

        dfs = []
        for i, name in enumerate(names):
            print("\rFirst Name: {0} {1}/{2}".format(name, i, len(names)), end='')

            source_name_edit_distance_df = graph_edit_distance_results_df[graph_edit_distance_results_df["Edit_Distance_Source_Name"] == name]
            source_name_soundex_df = target_sound_measure_results_df[target_sound_measure_results_df["{0}_Source_Name".format(self._targeted_sound_measure)] == name]

            #source_name_edit_distance_df = source_name_edit_distance_df.reset_index()
            #source_name_soundex_df = source_name_soundex_df.reset_index()

            if source_name_edit_distance_df.empty:
                prefix = "Edit_Distance"
                #column_names = list(source_name_edit_distance_df.columns)
                #source_name_soundex_df = source_name_soundex_df.reset_index()
                df_index = source_name_soundex_df.index.values[0]
                soundex_performance_tuple = source_name_soundex_df.ix[df_index]
                num_of_relevant_in_ground_truth = soundex_performance_tuple[4]

                source_name_edit_distance_df.set_value(df_index, "{0}_Source_Name".format(prefix), name)
                source_name_edit_distance_df.set_value(df_index, "{0}_Num of Relevant Retrieved".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_Num of Retrieved".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_Total Num of Relevant in Ground Truth".format(prefix), num_of_relevant_in_ground_truth)
                source_name_edit_distance_df.set_value(df_index, "{0}_Accuracy".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_F1".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_Precision".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_Precision_Calculated_by_Me".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_Recall".format(prefix), 0)
                source_name_edit_distance_df.set_value(df_index, "{0}_Recall_Calculated_by_Me".format(prefix), 0)

            if source_name_soundex_df.empty:
                # column_names = list(source_name_edit_distance_df.columns)
                # source_name_soundex_df = source_name_soundex_df.reset_index()
                df_index = source_name_edit_distance_df.index.values[0]
                soundex_performance_tuple = source_name_edit_distance_df.ix[df_index]
                num_of_relevant_in_ground_truth = soundex_performance_tuple[4]

                source_name_soundex_df.set_value(df_index, "{0}_Source_Name".format(self._targeted_sound_measure), name)
                source_name_soundex_df.set_value(df_index, "{0}_Num of Relevant Retrieved".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index, "{0}_Num of Retrieved".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index,
                                                       "{0}_Total Num of Relevant in Ground Truth".format(self._targeted_sound_measure),
                                                       num_of_relevant_in_ground_truth)
                source_name_soundex_df.set_value(df_index, "{0}_Accuracy".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index, "{0}_F1".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index, "{0}_Precision".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index, "{0}_Precision_Calculated_by_Me".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index, "{0}_Recall".format(self._targeted_sound_measure), 0)
                source_name_soundex_df.set_value(df_index, "{0}_Recall_Calculated_by_Me".format(self._targeted_sound_measure), 0)
                # column_names = list(source_name_soundex_df.columns)
                # source_name_edit_distance_df = source_name_edit_distance_df.reset_index()
                # performance_tuple = source_name_edit_distance_df.ix[0]
                # num_of_relevant_in_ground_truth = performance_tuple[4]
                # result_tuple = (name, 0, 0, num_of_relevant_in_ground_truth, 0, 0, 0, 0, 0, 0)
                # source_name_soundex_df = pd.DataFrame([result_tuple], columns=column_names)

            united_df = source_name_edit_distance_df.reset_index(drop=True).merge(source_name_soundex_df.reset_index(drop=True),
                                                                      left_index=True, right_index=True)
            dfs.append(united_df)

        results_df = pd.concat(dfs)

        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def unite_performance_of_name_based_network_and_rival(self):

        name_based_networks_results_df = pd.read_csv(self._input_directory_path + "name_based_network_ED_1_to_3_suggestions_with_ground_truth_results_performance.csv")
        target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_name_suggestions_by_soundex_with_ground_truth_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_name_suggestions_by_metaphone_with_ground_truth_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_name_suggestions_by_nysiis_with_ground_truth_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_name_suggestions_by_match_rating_codex_with_ground_truth_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_name_suggestions_by_double_metaphone_with_ground_truth_performance.csv")

        name_based_networks_df_columns = name_based_networks_results_df.columns
        name_based_networks_results_df = self._rename_columns(name_based_networks_df_columns, name_based_networks_results_df, "Name_Based_Networks")

        target_sound_measure_results_df_columns = target_sound_measure_results_df.columns
        target_sound_measure_results_df = self._rename_columns(target_sound_measure_results_df_columns,
                                                               target_sound_measure_results_df, self._targeted_sound_measure)

        #dfs = []

        united_df = name_based_networks_results_df.reset_index(drop=True).merge(target_sound_measure_results_df.reset_index(drop=True),
                                                                  left_index=True, right_index=True)
        #dfs.append(united_df)

        #results_df = pd.concat(dfs)

        united_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def _rename_columns(self, columns, df, targeted_string):
        for column in columns:
            df = df.rename(columns={column: '{0}_{1}'.format(column, targeted_string)})
        return df

    def unite_primary_and_secondary_to_single_double_metaphone(self):
        double_metaphone_primary_df = pd.read_csv(
            self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_primary.csv")

        double_metaphone_secondary_df = pd.read_csv(
            self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_secondary.csv")

        double_metaphone_primary_source_names_series = double_metaphone_primary_df["Source_Name"]
        double_metaphone_primary_source_names = double_metaphone_primary_source_names_series.tolist()

        double_metaphone_secondary_source_names_series = double_metaphone_secondary_df["Source_Name"]
        double_metaphone_secondary_source_names = double_metaphone_secondary_source_names_series.tolist()

        double_metaphone_source_names = double_metaphone_primary_source_names + double_metaphone_secondary_source_names

        source_names = list(set(double_metaphone_source_names))
        source_names = sorted(source_names)
        dfs = []
        for i, source_name in enumerate(source_names):
            print("\rFirst Name: {0} {1}/{2}".format(source_name, i, len(source_names)), end='')

            source_name_double_metaphone_primary_df = double_metaphone_primary_df[double_metaphone_primary_df["Source_Name"] == source_name]
            if not source_name_double_metaphone_primary_df.empty:
                dfs.append(source_name_double_metaphone_primary_df)

            source_name_double_metaphone_secondary_df = double_metaphone_secondary_df[double_metaphone_secondary_df["Source_Name"] == source_name]
            if not source_name_double_metaphone_secondary_df.empty:
                dfs.append(source_name_double_metaphone_secondary_df)

        results_df = pd.concat(dfs)

        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def unite_double_metaphone_primary_and_secondary_suggestions(self):
        double_metaphone_primary_df = pd.read_csv(
            self._input_directory_path + "first_name_suggestions_by_primary_metaphone.csv")

        double_metaphone_secondary_df = pd.read_csv(
            self._input_directory_path + "first_name_suggestions_by_secondary_metaphone.csv")

        double_metaphone_primary_df = double_metaphone_primary_df.rename(columns={'Double_Metaphone_Primary': 'Double_Metaphone'})
        double_metaphone_secondary_df = double_metaphone_secondary_df.rename(columns={'Double_Metaphone_Secondary': 'Double_Metaphone'})

        double_metaphone_primary_source_names_series = double_metaphone_primary_df["Original"]
        double_metaphone_primary_source_names = double_metaphone_primary_source_names_series.tolist()

        double_metaphone_secondary_source_names_series = double_metaphone_secondary_df["Original"]
        double_metaphone_secondary_source_names = double_metaphone_secondary_source_names_series.tolist()

        double_metaphone_source_names = double_metaphone_primary_source_names + double_metaphone_secondary_source_names

        source_names = list(set(double_metaphone_source_names))
        source_names = sorted(source_names)
        dfs = []
        for i, source_name in enumerate(source_names):
            print("\rFirst Name: {0} {1}/{2}".format(source_name, i, len(source_names)), end='')

            source_name_double_metaphone_primary_df = double_metaphone_primary_df[
                double_metaphone_primary_df["Original"] == source_name]

            source_name_double_metaphone_secondary_df = double_metaphone_secondary_df[
                double_metaphone_secondary_df["Original"] == source_name]



            double_metaphone_primary_num_of_rows = source_name_double_metaphone_primary_df.shape[0]
            double_metaphone_secondary_num_of_rows = source_name_double_metaphone_secondary_df.shape[0]
            if double_metaphone_secondary_num_of_rows == 10:
                double_metaphone_primary_num_of_rows = 5
                double_metaphone_secondary_num_of_rows = 5

            elif double_metaphone_secondary_num_of_rows < 5:
                double_metaphone_primary_num_of_rows = 10 - double_metaphone_secondary_num_of_rows



            source_name_double_metaphone_primary_first_five_df = source_name_double_metaphone_primary_df.head(double_metaphone_primary_num_of_rows)
            source_name_double_metaphone_secondary_first_five_df = source_name_double_metaphone_secondary_df.head(double_metaphone_secondary_num_of_rows)

            dfs.append(source_name_double_metaphone_primary_first_five_df)
            dfs.append(source_name_double_metaphone_secondary_first_five_df)


        results_df = pd.concat(dfs)

        results_df = results_df.sort_values(by="Original")
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def unite_double_metaphone_primary_and_secondary_suggestions_with_edit_distance(self):
        double_metaphone_primary_df = pd.read_csv(
            self._input_directory_path + "first_name_suggestions_by_primary_double_metaphone.csv")

        double_metaphone_secondary_df = pd.read_csv(
            self._input_directory_path + "first_name_suggestions_by_secondary_double_metaphone.csv")

        double_metaphone_primary_df = double_metaphone_primary_df.rename(columns={'Double_Metaphone_Primary': 'Double_Metaphone'})
        double_metaphone_secondary_df = double_metaphone_secondary_df.rename(columns={'Double_Metaphone_Secondary': 'Double_Metaphone'})

        double_metaphone_united_df = double_metaphone_primary_df.append(double_metaphone_secondary_df)

        double_metaphone_source_names_series = double_metaphone_united_df["Original"]
        double_metaphone_source_names = double_metaphone_source_names_series.tolist()

        source_names = list(set(double_metaphone_source_names))
        source_names = sorted(source_names)
        dfs = []
        for i, source_name in enumerate(source_names):
            print("\rFirst Name: {0} {1}/{2}".format(source_name, i, len(source_names)), end='')

            source_name_double_metaphone_df = double_metaphone_united_df[
                double_metaphone_united_df["Original"] == source_name]

            source_name_double_metaphone_df['Edit_Distance'] = source_name_double_metaphone_df.apply(
                lambda x: calculate_edit_distance(source_name, x["Candidate"]),
                axis=1)

            source_name_double_metaphone_df = source_name_double_metaphone_df.sort_values(by='Edit_Distance')

            # Take only first 10 for each name
            source_name_double_metaphone_df = source_name_double_metaphone_df.head(10)

            source_name_double_metaphone_df["Source_Name"] = source_name

            first_name_synonyms_by_soundex_df = source_name_double_metaphone_df[
                ['Original', 'Candidate', self._targeted_sound_measure]]
            # first_name_synonyms_by_soundex_df = first_name_synonyms_by_soundex_df.rename(
            #     columns={'Source_Name': 'Original', 'Name': 'Candidate'})

            dfs.append(first_name_synonyms_by_soundex_df)

        results_df = pd.concat(dfs)
        results_df = results_df.sort_values(by='Original')
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")


    def unite_performance_results_given_two_methods2(self):

        graph_edit_distance_results_df = pd.read_csv(self._input_directory_path + "graph_first_names_synonyms_by_edit_distance_1_sorted_by_source_name_results_performance.csv")
        soundex_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_soundex_results_performance.csv")
        #target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_results_performance.csv")
        target_sound_measure_results_df = pd.read_csv(self._input_directory_path + "first_names_chosen_by_graph_edit_distance_synonyms_by_double_metaphone_secondary_results_performance.csv")

        # remove names that are not exist in the ground truth
        graph_edit_distance_results_df = graph_edit_distance_results_df[graph_edit_distance_results_df["Edit_Distance_Total Num of Relevant in Ground Truth"] != 0]
        target_sound_measure_results_df = target_sound_measure_results_df[target_sound_measure_results_df["{0}_Total Num of Relevant in Ground Truth".format(self._targeted_sound_measure)] != 0]

        graph_edit_distance_names_series = graph_edit_distance_results_df["Edit_Distance_Source_Name"]
        graph_edit_distance_names = graph_edit_distance_names_series.tolist()

        soundex_names_series = target_sound_measure_results_df["{0}_Source_Name".format(self._targeted_sound_measure)]
        soundex_names = soundex_names_series.tolist()

        names = graph_edit_distance_names + soundex_names
        names = list(set(names))
        names = sorted(names)

        dfs = []
        for i, name in enumerate(names):
            print("\rFirst Name: {0} {1}/{2}".format(name, i, len(names)), end='')

            source_name_edit_distance_df = graph_edit_distance_results_df[graph_edit_distance_results_df["Edit_Distance_Source_Name"] == name]
            source_name_soundex_df = target_sound_measure_results_df[target_sound_measure_results_df["{0}_Source_Name".format(self._targeted_sound_measure)] == name]

            united_df = source_name_edit_distance_df.reset_index(drop=True).merge(source_name_soundex_df.reset_index(drop=True),
                                                                      left_index=True, right_index=True)
            dfs.append(united_df)

        results_df = pd.concat(dfs)

        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def unite_name_network_and_rival(self):
        double_metaphone_primary_df = pd.read_csv(
            self._input_directory_path + "name_based_network_ED_1_to_3_suggestions_with_ground_truth_results_performance.csv")

        double_metaphone_secondary_df = pd.read_csv(
            self._input_directory_path + "first_name_suggestions_by_soundex_with_ground_truth_results_performance.csv")

        double_metaphone_primary_source_names_series = double_metaphone_primary_df["Source_Name"]
        double_metaphone_primary_source_names = double_metaphone_primary_source_names_series.tolist()

        double_metaphone_secondary_source_names_series = double_metaphone_secondary_df["Source_Name"]
        double_metaphone_secondary_source_names = double_metaphone_secondary_source_names_series.tolist()

        double_metaphone_source_names = double_metaphone_primary_source_names + double_metaphone_secondary_source_names

        source_names = list(set(double_metaphone_source_names))
        source_names = sorted(source_names)
        dfs = []
        for i, source_name in enumerate(source_names):
            print("\rFirst Name: {0} {1}/{2}".format(source_name, i, len(source_names)), end='')

            source_name_double_metaphone_primary_df = double_metaphone_primary_df[double_metaphone_primary_df["Source_Name"] == source_name]
            if not source_name_double_metaphone_primary_df.empty:
                dfs.append(source_name_double_metaphone_primary_df)

            source_name_double_metaphone_secondary_df = double_metaphone_secondary_df[double_metaphone_secondary_df["Source_Name"] == source_name]
            if not source_name_double_metaphone_secondary_df.empty:
                dfs.append(source_name_double_metaphone_secondary_df)

        results_df = pd.concat(dfs)

        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

        print("Done!!!!!")

    def suggest_names_by_baseline_string_similarity(self):
        self._name_candidate_edit_distance_df = pd.DataFrame()

        dfs = []
        name_synonym_ground_truth_df = pd.read_csv(
            self._input_directory_path + self._name_synonym_ground_truth_file)

        # child_first_name_series = graph_first_name_edit_distance_1_df["Source_Name"]
        name_series = name_synonym_ground_truth_df["Name"]
        names = name_series.tolist()

        names = list(set(names))
        original_names = [name for name in names if len(name) > 2]
        original_names = sorted(original_names)
        # sorted_names_df = pd.DataFrame(sorted_names, columns=['Name:'])
        # sorted_names_df.to_csv(
        #     self._output_directory_path + "sorted_names_first_names_child_ancestor_count_greater_than_1_aggregated.csv",
        #     index=False)

        all_distinct_names_df = pd.read_csv(
            self._input_directory_path + self._target_file_name)

        distinct_names_series = all_distinct_names_df["Name"]
        distinct_names = distinct_names_series.tolist()
        distinct_names = sorted(distinct_names)

        for i, original_name in enumerate(original_names):
            distance_tuples = []
            #print("\rFirst Name: {0} {1}/{2}".format(name, i, len(original_names)), end='')
            for j, candidate in enumerate(distinct_names):
                print("\rFirst Name: {0} {1}/{2}, Candidate:{3} {4}/{5}".format(original_name, i, len(original_names), candidate, j, len(distinct_names)), end='')

                distance = globals()[self._selected_string_similarity_function](original_name, candidate)
                #distance = getattr(globals(), self._selected_string_similarity_function)(original_name, candidate)
                if distance > 0 and distance < 4:
                    distance_tuple = (original_name, candidate, distance)
                    distance_tuples.append(distance_tuple)

            df = pd.DataFrame(distance_tuples, columns=['Original', 'Candidate', self._targeted_sound_measure])
            df = df.sort_values(by=self._targeted_sound_measure)
            dfs.append(df)

            if i % 10000 == 0:
                results_df = pd.concat(dfs)
                #results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)
                results_df.to_csv(self._output_directory_path + "suggesting_names_by_{0}.csv".format(
                    self._selected_string_similarity_function), index=False)
                del results_df


        results_df = pd.concat(dfs)
        results_df.to_csv(self._output_directory_path + "suggesting_names_by_{0}.csv".format(self._selected_string_similarity_function), index=False)
        print("Done!!!!!")
        return results_df, name_synonym_ground_truth_df

    def suggest_names_by_sound_measure_compare_with_ground_truth_and_calculate_performance(self):
        suggestions_df, ground_truth_df = self.create_rival_sound_measure_suggestions()
        self._compare_suggestions_and_calculate_performance(suggestions_df, ground_truth_df)

    def suggest_names_by_baseline_similarity_function_compare_with_ground_truth_and_calculate_performance(self):
        suggestions_df, ground_truth_df = self.suggest_names_by_baseline_string_similarity()
        self._compare_suggestions_and_calculate_performance(suggestions_df, ground_truth_df)

    def _compare_suggestions_and_calculate_performance(self, suggestions_df, ground_truth_df):
        suggestions_with_ground_truth_df = self._compare_suggestions_with_ground_truth_by_provided_dfs(suggestions_df,
                                                                                                       ground_truth_df)
        self._calculate_performance(suggestions_with_ground_truth_df, ground_truth_df)


def calculate_edit_distance(name1, name2):
    if not name1 or not name2:
        return -1

    name1 = name1.lower()
    name2 = name2.lower()

    edit_dist = editdistance.eval(name1, name2)
    return edit_dist


def damerau_levenshtein_similarity(name1, name2):
    if not name1 or not name2:
        return -1

    name1 = str(name1)
    name2 = str(name2)

    damerau_levenshtein_distance = jellyfish.damerau_levenshtein_distance(name1, name2)
    return damerau_levenshtein_distance

# name1_name2_edit_distance = {}
# def calculate_edit_distance(name1, name2):
#     if not name1 or not name2:
#         return -1
#
#     name1 = name1.lower()
#     name2 = name2.lower()
#
#     key = name1 + " -> " + name2
#     opposite_key = name2 + " -> " + name1
#
#     if key not in name1_name2_edit_distance:
#         edit_dist = editdistance.eval(name1, name2)
#         name1_name2_edit_distance[key] = edit_dist
#         name1_name2_edit_distance[opposite_key] = edit_dist
#
#     return name1_name2_edit_distance[key]

def compare_suggestion(original_name, candidate, ground_truth_df):
    result_df = ground_truth_df[
        (ground_truth_df["Name"] == original_name) &
        (ground_truth_df["Synonym"] == candidate)]

    if result_df.empty:
        return 0
    return 1
