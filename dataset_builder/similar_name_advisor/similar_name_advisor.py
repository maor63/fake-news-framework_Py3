
from commons.method_executor import Method_Executor
from sframe import SFrame
import time
import pandas as pd
import editdistance
import networkx as nx
from sklearn.metrics import precision_score, accuracy_score, recall_score, precision_recall_fscore_support, f1_score
import phonetics

__author__ = "Aviad Elyashar"


class SimilarNameAdvisor(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_directory_path = self._config_parser.eval(self.__class__.__name__, "input_directory_path")
        self._target_file_name = self._config_parser.eval(self.__class__.__name__, "target_file_name")
        self._ground_truth_file_name = self._config_parser.eval(self.__class__.__name__, "ground_truth_file_name")
        self._output_directory_path = self._config_parser.eval(self.__class__.__name__, "output_directory_path")
        self._results_file_name = self._config_parser.eval(self.__class__.__name__, "results_file_name")
        self._ranking_function = self._config_parser.eval(self.__class__.__name__, "ranking_function")

    def _read_csv_file(self, delimiter="\t"):
        print("Reading CSV file ")
        begin_time = time.time()
        # for Micky's big file
        # wikitree_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter="\t")
        wikitree_sf = SFrame.read_csv(self._input_directory_path + self._target_file_name, delimiter=delimiter)
        end_time = time.time()
        run_time = end_time - begin_time
        print(run_time)
        return wikitree_sf


    def create_distinct_name_list(self):
        names_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        child_first_name_series = names_df["Child_First_Name"]
        child_first_names = child_first_name_series.tolist()

        ancestor_first_name_series = names_df["Ancestor_First_Name"]
        ancestor_first_names = ancestor_first_name_series.tolist()

        all_names = child_first_names + ancestor_first_names
        all_names = list(set(all_names))
        sorted_names = [name for name in all_names if len(name) > 1]
        sorted_names = sorted(sorted_names)
        sorted_names_df = pd.DataFrame(sorted_names, columns=['Name:'])
        sorted_names_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

    def provide_suggestions(self):
        ground_truth_df = pd.read_csv(self._input_directory_path + self._ground_truth_file_name)
        original_names_series = ground_truth_df["Name"]
        original_names = original_names_series.tolist()
        original_names = list(set(original_names))

        original_names = sorted(original_names)

        edges_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        name_graph = nx.from_pandas_edgelist(edges_df, 'Ancestor_First_Name',
                                             'Child_First_Name', ['count'])

        dfs = []

        for i, original_name in enumerate(original_names):
            print("\rSuggesting candidates for: {0} {1}/{2}".format(original_name, i, len(original_names)), end='')

            if name_graph.has_node(original_name):
                #candidates_df = self._suggest_names_for_original_name(name_graph, original_name)
                candidates_df = getattr(self, self._ranking_function)(name_graph, original_name)
                dfs.append(candidates_df)

        results_df = pd.concat(dfs)
        results_df = results_df.sort_values(by=['Original', 'Rank'])
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)
        print("Done provide_suggestions!!!!!")
        return results_df, ground_truth_df


    def provide_suggestions_save_with_constant_name(self):
        ground_truth_df = pd.read_csv(self._input_directory_path + self._ground_truth_file_name)
        original_names_series = ground_truth_df["Name"]
        original_names = original_names_series.tolist()
        original_names = list(set(original_names))

        original_names = sorted(original_names)

        edges_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        name_graph = nx.from_pandas_edgelist(edges_df, 'Ancestor_First_Name',
                                             'Child_First_Name', ['count'])

        dfs = []

        for i, original_name in enumerate(original_names):
            print("\rSuggesting candidates for: {0} {1}/{2}".format(original_name, i, len(original_names)), end='')

            if name_graph.has_node(original_name):
                #candidates_df = self._suggest_names_for_original_name(name_graph, original_name)
                candidates_df = getattr(self, self._ranking_function)(name_graph, original_name)
                dfs.append(candidates_df)

        results_df = pd.concat(dfs)
        results_df = results_df.sort_values(by=['Original', 'Rank'])
        results_df.to_csv(self._output_directory_path + "suggestions.csv", index=False)
        print("Done provide_suggestions!!!!!")
        return results_df, ground_truth_df

    #
    # Recieve the graph of father and son edit distance 1 until 3.
    # The ranking is according to double metaphone from the original name with edit distance.
    #
    def suggest_name_based_graph_ranking_by_minimal_edit_distance_of_double_metaphone(self, name_graph, original_name):
        nodes = nx.single_source_shortest_path_length(name_graph, original_name, 3)
        nodes = list(nodes.items())

        original_name_series = [original_name] * len(nodes)
        candidates_df = pd.DataFrame(nodes, columns=['Candidate', 'Order'])
        candidates_df['Original'] = original_name_series
        candidates_df = candidates_df[['Original', 'Candidate', 'Order']]

        candidates_df = candidates_df[candidates_df["Order"] != 0]
        candidates_df['Double_Metaphone_Primary_Original_Name'], \
        candidates_df['Double_Metaphone_Secondary_Original_Name'] = list(zip(*candidates_df.apply(
            lambda x: get_phonetics_double_metaphone(x["Original"]),
            axis=1)))
        candidates_df['Double_Metaphone_Primary_Candidate'], candidates_df['Double_Metaphone_Secondary_Candidate'] = list(zip(*candidates_df.apply(
            lambda x: get_phonetics_double_metaphone(x["Candidate"]),
            axis=1)))
        candidates_df["Edit_Distance_Primary_DM_Original_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Primary_Original_Name"], x["Double_Metaphone_Primary_Candidate"]),
            axis=1)
        candidates_df["Edit_Distance_Secondary_DM_Original_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Secondary_Original_Name"], x["Double_Metaphone_Secondary_Candidate"]),
                                axis=1)

        candidates_df["Edit_Distance_Primary_DM_Original_Secondary_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Primary_Original_Name"],
                                                                  x["Double_Metaphone_Secondary_Candidate"]),
                                axis=1)
        candidates_df["Edit_Distance_Secondary_DM_Original_Primary_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Secondary_Original_Name"],
                                                                  x["Double_Metaphone_Primary_Candidate"]),
                                axis=1)
        #candidates_df.to_csv(self._output_directory_path + "Metaphone_Edit_distance_graph.csv")
        candidates_df["Min_Edit_Distance_of_DM"] = \
            candidates_df.apply(lambda x: find_positive_min_value(x["Edit_Distance_Primary_DM_Original_Candidate"],
                                                                  x["Edit_Distance_Secondary_DM_Original_Candidate"],
                                                                  x["Edit_Distance_Primary_DM_Original_Secondary_Candidate"],
                                                                  x["Edit_Distance_Secondary_DM_Original_Primary_Candidate"]),
                                axis=1)

        candidates_df["Rank"] = candidates_df["Min_Edit_Distance_of_DM"]
        #candidates_df = candidates_df.sort_values(by='Min_Edit_Distance_of_DM')
        candidates_df = candidates_df.sort_values(by='Rank')
        #head_candidates_df = candidates_df.head(10)
        return candidates_df

    #
    # Order^2 *  Edit Distance
    #
    def suggest_names_by_order_2_and_ED(self, name_graph, original_name):
        nodes = nx.single_source_shortest_path_length(name_graph, original_name, 3)
        nodes = list(nodes.items())

        original_name_series = [original_name] * len(nodes)
        candidates_df = pd.DataFrame(nodes, columns=['Candidate', 'Order'])
        candidates_df['Original'] = original_name_series
        candidates_df = candidates_df[['Original', 'Candidate', 'Order']]

        candidates_df = candidates_df[candidates_df["Order"] != 0]

        candidates_df['Edit_Distance'] = candidates_df.apply(lambda x: calculate_edit_distance(x["Original"], x["Candidate"]),
                                                   axis=1)
        candidates_df['Shortest_Path'] = candidates_df.apply(
            lambda x: calculate_shortest_path(x["Original"], x["Candidate"], name_graph), axis=1)
        candidates_df['Rank'] = candidates_df.apply(lambda x: rank_candidate(x["Edit_Distance"], x["Order"], x["Shortest_Path"]),
                                          axis=1)

        candidates_df = candidates_df.sort_values(by='Rank')
        #head_candidates_df = candidates_df.head(10)
        #return head_candidates_df
        return candidates_df

    # Order * Edit Distance
    def suggest_name_by_edit_distance_and_order(self, name_graph, original_name):
        nodes = nx.single_source_shortest_path_length(name_graph, original_name, 3)
        nodes = list(nodes.items())

        original_name_series = [original_name] * len(nodes)
        candidates_df = pd.DataFrame(nodes, columns=['Candidate', 'Order'])
        candidates_df['Original'] = original_name_series
        candidates_df = candidates_df[['Original', 'Candidate', 'Order']]

        candidates_df = candidates_df[candidates_df["Order"] != 0]

        candidates_df['Edit_Distance'] = candidates_df.apply(lambda x: calculate_edit_distance(x["Original"], x["Candidate"]),
                                                   axis=1)
        candidates_df['Rank'] = candidates_df.apply(lambda x: rank_candidate_ED_and_order(x["Edit_Distance"], x["Order"]),
                                          axis=1)

        candidates_df = candidates_df.sort_values(by='Rank')
        #head_candidates_df = candidates_df.head(10)
        #return head_candidates_df
        return candidates_df


        # Order * Edit Distance
    def suggest_name_by_edit_distance_and_order_and_ED_of_DM(self, name_graph, original_name):
        nodes = nx.single_source_shortest_path_length(name_graph, original_name, 3)
        nodes = list(nodes.items())

        original_name_series = [original_name] * len(nodes)
        candidates_df = pd.DataFrame(nodes, columns=['Candidate', 'Order'])
        candidates_df['Original'] = original_name_series
        candidates_df = candidates_df[['Original', 'Candidate', 'Order']]

        candidates_df = candidates_df[candidates_df["Order"] != 0]

        candidates_df['Edit_Distance'] = candidates_df.apply(
            lambda x: calculate_edit_distance(x["Original"], x["Candidate"]),
            axis=1)

        candidates_df['Double_Metaphone_Primary_Original_Name'], \
        candidates_df['Double_Metaphone_Secondary_Original_Name'] = list(zip(*candidates_df.apply(
            lambda x: get_phonetics_double_metaphone(x["Original"]),
            axis=1)))
        candidates_df['Double_Metaphone_Primary_Candidate'], candidates_df['Double_Metaphone_Secondary_Candidate'] = list(zip(*candidates_df.apply(
            lambda x: get_phonetics_double_metaphone(x["Candidate"]),
            axis=1)))
        candidates_df["Edit_Distance_Primary_DM_Original_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Primary_Original_Name"], x["Double_Metaphone_Primary_Candidate"]),
            axis=1)
        candidates_df["Edit_Distance_Secondary_DM_Original_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Secondary_Original_Name"], x["Double_Metaphone_Secondary_Candidate"]),
                                axis=1)

        candidates_df["Edit_Distance_Primary_DM_Original_Secondary_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Primary_Original_Name"],
                                                                  x["Double_Metaphone_Secondary_Candidate"]),
                                axis=1)
        candidates_df["Edit_Distance_Secondary_DM_Original_Primary_Candidate"] = \
            candidates_df.apply(lambda x: calculate_edit_distance(x["Double_Metaphone_Secondary_Original_Name"],
                                                                  x["Double_Metaphone_Primary_Candidate"]),
                                axis=1)
        #candidates_df.to_csv(self._output_directory_path + "Metaphone_Edit_distance_graph.csv")
        candidates_df["Min_Edit_Distance_of_DM"] = \
            candidates_df.apply(lambda x: find_positive_min_value(x["Edit_Distance_Primary_DM_Original_Candidate"],
                                                                  x["Edit_Distance_Secondary_DM_Original_Candidate"],
                                                                  x["Edit_Distance_Primary_DM_Original_Secondary_Candidate"],
                                                                  x["Edit_Distance_Secondary_DM_Original_Primary_Candidate"]),
                                axis=1)


        candidates_df['Rank'] = candidates_df.apply(
            lambda x: rank_candidate_ED_and_order_and_ED_of_DM(x["Edit_Distance"], x["Order"], x["Min_Edit_Distance_of_DM"]), axis=1)

        candidates_df = candidates_df.sort_values(by='Rank')
        #head_candidates_df = candidates_df.head(10)
        #return head_candidates_df
        return candidates_df

    def compare_suggestions_with_ground_truth(self):
        ground_truth_df = pd.read_csv(self._input_directory_path + self._ground_truth_file_name)
        suggestions_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        suggestions_with_ground_truth_df = self._compare_suggestion_with_ground_truth_by_provided_dfs(suggestions_df, ground_truth_df)
        return suggestions_with_ground_truth_df

    def _compare_suggestion_with_ground_truth_by_provided_dfs(self, suggestions_df, ground_truth_df):
        suggestions_df['Is_Original_Synonym'] = suggestions_df.apply(
            lambda x: compare_suggestion(x["Original"], x["Candidate"], ground_truth_df),
            axis=1)

        suggestions_df.to_csv(self._output_directory_path + "suggestions_with_ground_truth.csv", index=False)
        return suggestions_df



    def calculate_performance_for_suggestions(self):
        suggestions_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        ground_truth_df = pd.read_csv(self._input_directory_path + self._ground_truth_file_name)
        self._calculate_performance(suggestions_df, ground_truth_df)


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
        final_results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

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

    def create_suggestion_find_hits_in_ground_truth_and_calculate_performance(self):
        suggestions_df, ground_truth_df = self.provide_suggestions_save_with_constant_name()
        suggestions_with_ground_truth_df = self._compare_suggestion_with_ground_truth_by_provided_dfs(suggestions_df, ground_truth_df)
        self._calculate_performance(suggestions_with_ground_truth_df, ground_truth_df)



def compare_suggestion(original_name, candidate, ground_truth_df):
    result_df = ground_truth_df[
        (ground_truth_df["Name"] == original_name) &
        (ground_truth_df["Synonym"] == candidate)]

    if result_df.empty:
        return 0
    return 1


def calculate_shortest_path(original_name, candidate, graph):
    shortest_path = nx.shortest_path_length(graph, source=original_name, target=candidate)
    return shortest_path

def rank_candidate(edit_distance_result, order, shortest_path):
    rank = edit_distance_result * order * shortest_path
    return rank

def rank_candidate_ED_and_order(edit_distance_result, order):
    rank = edit_distance_result * order
    return rank

def rank_candidate_ED_and_order_and_ED_of_DM(edit_distance_result, order, min_edit_distance_of_DM):
    rank = edit_distance_result * order * (min_edit_distance_of_DM + 1)
    return rank

name1_name2_edit_distance = {}
def calculate_edit_distance(name1, name2):
    if not name1 or not name2:
        return -1

    name1 = name1.lower()
    name2 = name2.lower()

    key = name1 + " -> " + name2
    opposite_key = name2 + " -> " + name1

    if key not in name1_name2_edit_distance:
        edit_dist = editdistance.eval(name1, name2)
        name1_name2_edit_distance[key] = edit_dist
        name1_name2_edit_distance[opposite_key] = edit_dist

    return name1_name2_edit_distance[key]

def get_phonetics_double_metaphone(name):
    # if name is not None and name is not 'None' and name is not '':
    #     # name = unicode(name)
    result = phonetics.dmetaphone(name)
    return result[0], result[1]

def find_positive_min_value(value1, value2, value3, value4):
    array = [value1, value2, value3, value4]
    min_value = min(i for i in array if i >= 0)
    return min_value