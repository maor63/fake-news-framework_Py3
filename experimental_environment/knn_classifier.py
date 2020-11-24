from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd
import operator
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, accuracy_score, confusion_matrix
from itertools import product
import networkx as nx
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import csv
from collections import OrderedDict
from dataset_builder.feature_extractor.anchor_authors_creator import AnchorAuthorsCreator
from commons.consts import PerformanceMeasures
import math
import copy
from .results_container import ResultsContainer

class KNN_Classifier(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._results_path = self._config_parser.get(self.__class__.__name__, "results_path")
        self._predictions_per_iteration_path = self._config_parser.get(self.__class__.__name__, "predictions_per_iteration_path")
        self._similarity_functions = self._config_parser.eval(self.__class__.__name__, "similarity_functions")
        self._k = self._config_parser.eval(self.__class__.__name__, "k")
        self._compute_knn_based_on_link_prediction = self._config_parser.eval(self.__class__.__name__, "compute_knn_based_on_lp")
        self._link_prediction_models = self._config_parser.eval(self.__class__.__name__, "link_prediction_models")
        self._results_averaged_on_report = self._config_parser.eval(self.__class__.__name__, "results_averaged_on_report")
        self._generate_anchors = self._config_parser.eval(self.__class__.__name__, "generate_anchors")
        self._num_iterations = self._config_parser.eval(self.__class__.__name__, "num_iterations")
        self._targeted_class_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_dict")
        self._decision_models = self._config_parser.eval(self.__class__.__name__, "decision_models")
        self._targeted_class_field_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_name")
        self._order_of_results_dictionary = self._config_parser.eval(self.__class__.__name__, "order_of_results_dictionary")
        self._results_table_file_name = self._config_parser.get(self.__class__.__name__, "results_table_file_name")
        self._path = self._config_parser.get(self.__class__.__name__, "path")

        self._column_names_for_results_table = self._config_parser.eval(self.__class__.__name__, "column_names_for_results_table")
        self._index_field_for_predictions = self._config_parser.get(self.__class__.__name__, "index_field_for_predictions")

        self._targeted_class_anchors_percent_size = self._config_parser.eval(self.__class__.__name__, "targeted_class_anchors_percent_size")
        self._targeted_class_num_of_authors_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_num_of_authors_dict")


    def set_up(self):
        pass

    def execute(self, window_start=None):
        self._num_to_targeted_class_dict = {v: k for k, v in self._targeted_class_dict.items()}

        self._results_dict = self._create_results_dictionary()



        labeled_author_dict, unlabeled_author_dict, unlabeled_author_guid_index_field_dict = self._db.create_author_dictionaries(self._index_field_for_predictions, self._domain)
        for author_id, targeted_class in labeled_author_dict.items():
            labeled_author_dict[author_id] = self._targeted_class_dict[targeted_class]
        total_authors = len(labeled_author_dict)


        experiments = list(product(self._similarity_functions, self._k, self._targeted_class_anchors_percent_size, range(self._num_iterations)))
        report_lines = []
        prediction_csv_lines = []
        results = defaultdict(list)
        result_tuples = []
        for experiment in experiments:
            print(experiment)
            start_time = datetime.now()
            graph_name = experiment[0]
            k = experiment[1]
            anchors_percent_size = experiment[2]
            iteration = experiment[3]
            ###new part

            number_of_anchors = int(math.ceil(anchors_percent_size*total_authors))
            custom_targeted_class_num_of_authors_dict = dict.fromkeys(list(self._num_to_targeted_class_dict.values()), None)
            anchor_authors_creator = AnchorAuthorsCreator(self._db, custom_targeted_class_num_of_authors_dict)
            self._targeted_class_author_guids_dict = anchor_authors_creator.get_targeted_class_author_guids_dict()
            # get anchor-authors as training set
            train_set = []

            self._db.delete_anchor_authors()

            targeted_class_anchor_ids = {}
            for target_class in self._targeted_class_author_guids_dict:
                author_guids = self._targeted_class_author_guids_dict[target_class]
                num_of_author_guids = len(author_guids)
                number_of_anchors = int(math.ceil(anchors_percent_size * num_of_author_guids))
                anchor_author_guids = list(random.sample(author_guids, number_of_anchors))

                targeted_class_anchor_ids[target_class] = anchor_author_guids

            for target_class in targeted_class_anchor_ids:
                anchor_guids = targeted_class_anchor_ids[target_class]
                for anchor_guid in anchor_guids:
                    train_set.append((anchor_guid, self._targeted_class_dict[target_class]))

            total_train_size = len(train_set)
            train_set = pd.DataFrame(train_set, columns=['id', 'class'])
            test_set = []
            num_train_nodes_in_graph = 0
            num_test_nodes_in_graph = 0
            number_of_egdes = 0


            title = 'KNN '
            '''KNN *without* LP, we want the neighbors (train set) with K largest weights to the test nodes.
               After finding the k train nodes with largest weights, look at their class and
               assign the majority class to the test node'''
            connections = self._db.get_labeled_author_connections_by_connection_type(graph_name)
            connections = pd.DataFrame(connections, columns=['source_id', 'destination_id', 'weight'])
            id_train = train_set.set_index(keys=['id']).index
            id_src_con = connections.set_index(keys=['source_id']).index
            id_dst_con = connections.set_index(keys=['destination_id']).index
            sources = connections[(~id_src_con.isin(id_train) & id_dst_con.isin(id_train))] #souce is test, and dest is train
            destinations = connections[(id_src_con.isin(id_train) & ~id_dst_con.isin(id_train))] #opposite from above
            test_set = set(sources['source_id'].tolist() + destinations['destination_id'].tolist()) #set of test nodes Ids
            num_test_nodes_in_graph = len(test_set)
            train_test_connections = sources.append(destinations) #connections between train and test nodes

            train_nodes = set(sources['destination_id'].tolist() + destinations['source_id'].tolist()) #set of train nodes Ids
            num_train_nodes_in_graph = len(train_nodes)
            number_of_egdes = len(train_test_connections)

            author_predictions = {}
            actual = {}
            for decision_model in self._decision_models:
                author_predictions[decision_model] = {}
                actual[decision_model] = {}

            counter = 0
            for test_set_author in test_set:
                if test_set_author not in labeled_author_dict:
                    continue
                counter += 1
                if self._compute_knn_based_on_link_prediction:
                    link_prediction_heuristic = experiment[2]
                    neighbors = self.get_link_prediction(test_set_author, train_set, graph, k, link_prediction_heuristic)
                else:
                    neighbors = self.get_neighbors(test_set_author, train_set, train_test_connections, k)

                #neighbors_types = [neighbor[1] for neighbor in neighbors]
                #confidence_level, prediction = self.majority_voting(neighbors)
                for decision_model in self._decision_models:
                    confidence_level, prediction = getattr(self, decision_model)(neighbors)
                    if prediction is not None:
                        author_predictions[decision_model][test_set_author] = prediction
                        actual[decision_model][test_set_author] = labeled_author_dict[test_set_author]
                    line = self.create_row_for_prediction_csv(iteration, graph_name, test_set_author, decision_model, confidence_level, self._num_to_targeted_class_dict[author_predictions[decision_model][test_set_author]], self._num_to_targeted_class_dict[actual[decision_model][test_set_author]],neighbors)
                    prediction_csv_lines.append(line)

            train_set = set(train_set['id'])#cast from series to set, for easier checking if element is in set
            '''Random classification for authors that are neither in training set nor appear in the graph is done here'''
            num_random_guesses = 0
            optional_values = list(self._targeted_class_dict.values())
            for author_osn_id in list(labeled_author_dict.keys()):
                if author_osn_id not in test_set and author_osn_id not in train_set:
                    random_classification = random.choice(optional_values)
                    for decision_model in self._decision_models:
                        author_predictions[decision_model][author_osn_id] = random_classification

                    actual_classification = labeled_author_dict[author_osn_id]
                    for decision_model in self._decision_models:
                        actual[decision_model][author_osn_id] = actual_classification
                    num_random_guesses += 1

            #author_predictions = [1 if author_predictions[x] == u'bad_actor' else 0 for x in author_predictions]
            for decision_model in self._decision_models:
                genuineness_predictions = []
                actuals = []
                genuineness_predictions = [author_predictions[decision_model][author_guid] for author_guid in author_predictions[decision_model]]
                #actual = [1 if actual[x] == u'bad_actor' else 0 for x in actual]
                actuals = [actual[decision_model][author_guid] for author_guid in actual[decision_model]]


                heuristic = decision_model

                end_time = datetime.now()
                duration = end_time - start_time


                if len(actual) > 0 and len(genuineness_predictions) > 0:
                    # report = classification_report(actual, predictions, target_names=[' 0 good actor', '1 bad actor'])
                    report = classification_report(actuals, genuineness_predictions)
                    print(report)

                    auc_score = roc_auc_score(actuals, genuineness_predictions)
                    accuracy = accuracy_score(actuals, genuineness_predictions)
                    f1 = f1_score(actuals, genuineness_predictions)
                    precision = precision_score(actuals, genuineness_predictions)
                    recall = recall_score(actuals, genuineness_predictions)
                    conf_matrix = confusion_matrix(actuals, genuineness_predictions)

                    result_tuple = (graph_name, k, anchors_percent_size, iteration, decision_model, auc_score, accuracy, f1, precision, recall, conf_matrix)
                    result_tuples.append(result_tuple)

        df = pd.DataFrame(result_tuples, columns=['Similarity Function', 'K', "%Training Size", "Decision", "#Iteration", "AUC", "Accuracy", "F1",
                                   "Precision", "Recall", "Confusion Matrix"])
        df.to_csv(self._path + "KNN_SNAM_Kaggle_Twitter_ISIS_Propaganda.csv", index=None)
        # self.save_results_to_csv(report_lines, results)
        # self.write_predictions_to_csv(prediction_csv_lines)
        #
        # self._result_container.calculate_average_performances(self._num_iterations)
        # self._result_container.write_results_as_table()
        #
        # selected_combination = self._result_container.find_max_average_auc_classifier()
        #
        # self._predict_on_unlabeled_authors_and_save(selected_combination, labeled_author_dict, unlabeled_author_guid_index_field_dict)



    def compute_results(self, heuristic, graph_name, iteration, k, predictions, actual, title,  duration, results, anchors_percent_size):
        print(title)
        if heuristic is not None:
            print('Link prediciton model: '+heuristic)
        print(' K: ' + str(k))
        print(' Similarity Function: ' + graph_name)
        print(' Iteration: '+str(iteration))
        print(' Duration: '+str(duration))
        if len(actual) > 0 and len(predictions) > 0:
            #report = classification_report(actual, predictions, target_names=[' 0 good actor', '1 bad actor'])
            report = classification_report(actual, predictions)
            print(report)
            print('AUC ' + str(roc_auc_score(actual, predictions)))
        else:
            print('No data')

        auc = roc_auc_score(actual, predictions)
        results[(k,'auc')] += [auc]
        targeted_class_field_name = self._targeted_class_field_name[0]
        #current_classifier = self._results_dict[targeted_class_field_name][graph_name][k][heuristic]

        self._result_container.set_result(auc, PerformanceMeasures.AUC, targeted_class_field_name, graph_name, k, heuristic, anchors_percent_size) #original
        #self._result_container.set_result(auc, PerformanceMeasures.AUC, targeted_class_field_name, graph_name, k, heuristic) #by Lior
        #current_classifier[PerformanceMeasures.AUC] += auc

        if not self._results_averaged_on_report:
            performance = precision_recall_fscore_support(actual, predictions)
            results[(k, 'precision')] += [float((performance[0][0] + performance[0][1]) / 2)]
            results[(k, 'recall')] += [float((performance[1][0] + performance[1][1]) / 2)]
            results[(k, 'f1')] += [float((performance[2][0] + performance[2][1]) / 2)]
            results[(k, 'support')] += [float((performance[3][0] + performance[3][1]) / 2)]

        else:
            performance = precision_recall_fscore_support(actual, predictions, average='binary')

            precision = performance[0]
            results[(k, 'precision')] += [precision]

            self._result_container.set_result(precision, PerformanceMeasures.PRECISION, targeted_class_field_name, graph_name, k,
                                              heuristic, anchors_percent_size) #original
            #self._result_container.set_result(precision, PerformanceMeasures.PRECISION, targeted_class_field_name,
            #                                  graph_name, k,
            #                                  heuristic) #By Lior

            recall = performance[1]
            results[(k, 'recall')] += [recall]
            self._result_container.set_result(recall, PerformanceMeasures.RECALL, targeted_class_field_name,
                                             graph_name, k,
                                            heuristic, anchors_percent_size) #original
            #self._result_container.set_result(recall, PerformanceMeasures.RECALL, targeted_class_field_name,
            #                                  graph_name, k,
            #                                  heuristic) #by Lior

            f1 = performance[2]
            results[(k, 'f1')] += [f1]
            results[(k, 'support')] = None

            accuracy = accuracy_score(actual, predictions)
            self._result_container.set_result(accuracy, PerformanceMeasures.ACCURACY, targeted_class_field_name,
                                              graph_name, k,
                                              heuristic, anchors_percent_size)
            # self._result_container.set_result(accuracy, PerformanceMeasures.ACCURACY, targeted_class_field_name,
            #                                   graph_name, k,
            #                                   heuristic)

            confusion_matrix_score = confusion_matrix(actual, predictions)
            self._result_container.set_result(confusion_matrix_score, PerformanceMeasures.CONFUSION_MATRIX, targeted_class_field_name,
                                              graph_name, k,
                                              heuristic, anchors_percent_size)
            # self._result_container.set_result(confusion_matrix_score, PerformanceMeasures.CONFUSION_MATRIX,
            #                                   targeted_class_field_name,
            #                                   graph_name, k,
            #                                   heuristic)
            num_of_correct_instances, num_of_incorrect_instances = self._result_container.calculate_correctly_and_not_correctly_instances(confusion_matrix_score)

            self._result_container.set_result(num_of_correct_instances, PerformanceMeasures.CORRECTLY_CLASSIFIED,
                                              targeted_class_field_name,
                                              graph_name, k,
                                              heuristic, anchors_percent_size)
            # self._result_container.set_result(num_of_correct_instances, PerformanceMeasures.CORRECTLY_CLASSIFIED,
            #                                   targeted_class_field_name,
            #                                   graph_name, k,
            #                                   heuristic)

            self._result_container.set_result(num_of_incorrect_instances, PerformanceMeasures.INCORRECTLY_CLASSIFIED,
                                              targeted_class_field_name,
                                              graph_name, k,
                                              heuristic, anchors_percent_size)
            # self._result_container.set_result(num_of_incorrect_instances, PerformanceMeasures.INCORRECTLY_CLASSIFIED,
            #                                   targeted_class_field_name,
            #                                   graph_name, k,
            #                                   heuristic)

            self._result_container.set_result(num_of_incorrect_instances, PerformanceMeasures.SELECTED_FEATURES,
                                              targeted_class_field_name,
                                              graph_name, k,
                                              heuristic, anchors_percent_size)

            # self._result_container.set_result(num_of_incorrect_instances, PerformanceMeasures.SELECTED_FEATURES,
            #                                   targeted_class_field_name,
            #                                   graph_name, k,
            #                                   heuristic)

    def save_results_to_csv(self, report_lines, all_iter_results):

        filename = self._results_path

        header = 'Total_Observations,Train_Size,Graph,Heuristic,K,Start_Date,End_Date,Duration,' \
                 'Train_Nodes_in_Graph,Test_Nodes_in_Graph,Number_of_Edges,Number_of_Random' \
                 'Guesses,Memory Consumption,Iteration'
        if not self._results_averaged_on_report:
            header += ', Precision Class 0, Precision Class 1, Recall Class 0, Recall Class 1, F1 Class 0, ' \
                      'F1 Class 1, Support Class 0, Support Class 1 '
        else:
            header += ',Precision,Recall,F1,Support,'

        header += 'AUC,Avg_Precision,Avg_Recall,Avg_F1,Avg_AUC,Avg_Support,Std_Dev_Precision,' \
                  'STD_Dev_Recall,Sev_Dev_F1,Std_Dev_AUC,Std_Dev_Support  \n'


        with open(filename, "w") as text_file:
            text_file.write(header)
            for iteration, total_authors, train_size, graph_name, heuristic, k, prediction, actual, start_time, end_time, duration,\
                    train_nodes_in_graph, test_nodes_in_graph, num_random_guesses, memory_usage, num_edges in report_lines:

                    line = str(total_authors) + ',' + str(train_size) + ',' + str(graph_name) + ',' + heuristic + ',' \
                           + str(k) + ',' + str(start_time) + ',' + str(end_time) + ',' + str(duration) + ',' \
                           + str(train_nodes_in_graph) + ',' + str(test_nodes_in_graph) + ',' \
                           + str(num_edges) + ',' + str(num_random_guesses) + ',' + str(memory_usage)+' ,'+str(iteration)

                    if not self._results_averaged_on_report:
                        results = precision_recall_fscore_support(actual, prediction)
                        prec_a = results[0][0]
                        prec_b = results[0][1]
                        rec_a = results[1][0]
                        rec_b = results[1][1]
                        f1_a = results[2][0]
                        f1_b = results[2][1]
                        support_a = results[3][0]
                        support_b = results[3][1]
                        line += ','+str(prec_a) + ',' + str(prec_b) + ',' + str(rec_a) + ',' + str(rec_b) \
                                + ',' + str(f1_a) + ', ' + str(f1_b) + ',' + str(support_a) + ',' + str(support_b)
                    else:
                        results = precision_recall_fscore_support(actual, prediction, average='binary')
                        prec = results[0]
                        rec = results[1]
                        f1 = results[2]
                        support = results[3]
                        line += ',' + str(prec) + ',' + str(rec) + ',' + str(f1) + ', ' + str(support)
                    auc = roc_auc_score(actual, prediction)

                    avg_precision = np.mean(all_iter_results[(k, 'precision')])
                    avg_recall = np.mean(all_iter_results[(k, 'recall')])
                    avg_f1 = np.mean(all_iter_results[(k, 'f1')])
                    avg_auc = np.mean(all_iter_results[(k, 'auc')])
                    if all_iter_results[(k, 'support')] is not None \
                            and len(all_iter_results[(k, 'support')]) > 0:
                        avg_support = np.mean(all_iter_results[(k, 'support')])
                        stddev_support = np.std(all_iter_results[(k, 'support')])
                    else:
                        avg_support = 0.0
                        stddev_support = 0.0

                    stddev_precision = np.std(all_iter_results[(k, 'precision')])
                    stddev_recall = np.std(all_iter_results[(k, 'recall')])
                    stddev_f1 = np.std(all_iter_results[(k, 'f1')])
                    stddev_auc = np.std(all_iter_results[(k, 'auc')])

                    line += ',' + str(auc) + ',' + str(avg_precision) + ',' + str(avg_recall) + ',' + str(avg_f1) + ', '\
                            + str(avg_auc) + ',' + str(avg_support) + ', '+str(stddev_precision)+', '+ str(stddev_recall)+', '+\
                            str(stddev_f1)+', '+str(stddev_auc)+','+str(stddev_support) + ' \n'

                    text_file.write(line)

    def get_neighbors(self, instance, train, connections, k):
        neighbors_are_source = connections.loc[connections['destination_id'] == instance][['source_id', 'weight']]
        neighbors_are_source = neighbors_are_source.rename(columns={'source_id': 'neighbor_id'})

        neighbors_are_dest = connections.loc[connections['source_id'] == instance][['destination_id','weight']]
        neighbors_are_dest = neighbors_are_dest.rename(columns={'destination_id': 'neighbor_id'})

        all_neighbors = neighbors_are_source.append(neighbors_are_dest)
        all_neighbors = all_neighbors.drop_duplicates()
        neighbors = self.filter_k_nearest_by_weight(all_neighbors, k, train)
        return neighbors

    def get_link_prediction(self, instance, train, graph, k, link_prediciton_model):
        link_coefficients = {}
        if instance in graph:
            for index, train_example in train.iterrows():
                train_example_id = train_example['id']
                if train_example_id not in graph:
                    continue
                pair = (instance, train_example_id)

                if link_prediciton_model == 'friends_measure' or link_prediciton_model=='jaccard_coefficient_weighted_sum' \
                    or link_prediciton_model == 'weighted_jaccard_coefficient_fuzzy_sets':
                    score = getattr(self._link_prediction_feature_extractor, link_prediciton_model)(graph, instance, train_example_id)
                    link_coefficients[train_example_id] = score
                else:
                    if link_prediciton_model == 'common_neighbors':
                        preds = nx.nx.cn_soundarajan_hopcroft(graph, [pair])
                        common_neighbors = list(preds)
                        common_neighbors_count = len(common_neighbors)
                        link_coefficients[train_example_id] = common_neighbors_count
                    else:
                        preds = getattr(nx, link_prediciton_model)(graph, [pair])
                        for u, v, p in preds:
                            link_coefficients[v] = p
        else:
            return pd.DataFrame()

        most_probable_links = pd.DataFrame(list(link_coefficients.items()), columns=['neighbor_id','weight'])
        most_probable_links = self.filter_k_nearest_by_weight(most_probable_links, k, train)
        return most_probable_links

    def filter_k_nearest_by_weight(self, all_neighbors, k, train):
        nearest_neighbors = all_neighbors.sort_values(by="weight", ascending=False).reset_index()

        num_of_neighbors = len(nearest_neighbors.index)
        if k <= num_of_neighbors:
            nearest_neighbors = nearest_neighbors.head(k)
        neighbors = []
        for index, neighbor in nearest_neighbors.iterrows():
            id = neighbor['neighbor_id']
            weight = neighbor['weight']
            target_class = train[train['id'] == id]['class'].values[0]
            neighbors.append((id, weight, target_class))
        return neighbors

    def majority_voting(self, neighbors):
        class_predictions, targeted_class_weights_dict = self._calculate_num_of_neighbors_and_weights_per_class_to_dicts(neighbors)
        sorted_neighbors_per_class = sorted(iter(class_predictions.items()), key=operator.itemgetter(1), reverse=True)
        if len(sorted_neighbors_per_class) == 0:
            return None, None
        majority_class = sorted_neighbors_per_class[0][0]
        majority_class_number = sorted_neighbors_per_class[0][1]
        total_neighbors = len(neighbors)
        confidence_level = majority_class_number / (total_neighbors * 1.0)
        if majority_class == 0:
            confidence_level = 1 - confidence_level
        return confidence_level, majority_class

    def weighted_majority_voting(self, neighbors):
        if len(neighbors) > 0:
            class_predictions, targeted_class_weights_dict = self._calculate_num_of_neighbors_and_weights_per_class_to_dicts(neighbors)
            confidence_level = 0
            targeted_class_average_weight_dict = {}
            for targeted_class, weights in targeted_class_weights_dict.items():
                if len(weights) > 0:
                    average_weight = sum(weights) / float(len(weights))
                    targeted_class_average_weight_dict[targeted_class] = average_weight
            sorted_targeted_class_average_weight_dict = sorted(iter(targeted_class_average_weight_dict.items()), key=operator.itemgetter(1), reverse=True)

            majority_class = sorted_targeted_class_average_weight_dict[0][0]
            averages = [average for targeted_class, average in targeted_class_average_weight_dict.items()]
            confidence_level = sum(averages) / float(len(averages))
        else:
            # if there are no neighbors - 50% to be good actor.
            confidence_level = 0.5
            majority_class = 0

        return confidence_level, majority_class

    def bad_actors_neighbors(self, neighbors):
        class_predictions, targeted_class_weights_dict = self._calculate_num_of_neighbors_and_weights_per_class_to_dicts(neighbors)
        bad_actor_class = 1
        if bad_actor_class in class_predictions:
            num_of_bad_actors = class_predictions[bad_actor_class]
            total_neighbors = len(neighbors)

            #confidence_level = 0.5 - (num_of_bad_actors / (total_neighbors * 1.0))
            confidence_level = num_of_bad_actors / (total_neighbors * 1.0)
            #changing the range from [-0.5,0.5] to [0,1]
            #confidence_level += 0.5

            if confidence_level > 0.5:
                # bad_actor
                return confidence_level, 1
                # good_actor
            return confidence_level, 0
        # No bad actors at all so confidence level = 0.5, and it is good actor.
        return 1, 0

    def _calculate_num_of_neighbors_and_weights_per_class_to_dicts(self, neighbors):
        class_predictions = {}
        targeted_class_weights_dict = {num: [] for num, targeted_class in self._num_to_targeted_class_dict.items()}
        for neighbor in neighbors:
            weight = neighbor[1]
            targeted_class = neighbor[2]
            #num_of_class = self._targeted_class_dict[targeted_class]
            if targeted_class in class_predictions:
                class_predictions[targeted_class] += 1
            else:
                class_predictions[targeted_class] = 1
            targeted_class_weights_dict[targeted_class].append(weight)
        return class_predictions, targeted_class_weights_dict

    def _calculate_weighted_num_of_neighbors_per_class(self, neighbors):
        class_predictions = {}
        for neighbor in neighbors:
            targeted_class = neighbor[2]
            if targeted_class in class_predictions:
                class_predictions[targeted_class] += 1
            else:
                class_predictions[targeted_class] = 1
        return class_predictions

    def memory_usage_psutil(self):
        # return the memory usage in MB
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(2 ** 20)
        return mem


    def write_predictions_to_csv(self,rows):
        with open(self._predictions_per_iteration_path, "wb") as text_file:
            header = OrderedDict([("iteration",None), ("graph_name",None), ("user_guid",None), ("decision_model",None), ("confidence_level",None), ("prediction",None),("actual",None),("neighbors",None)])
            dw = csv.DictWriter(text_file,header)
            dw.writeheader()
            dw.writerows(rows)

    def create_row_for_prediction_csv(self, itaretion, graph_name, user_guid, decision_model, confidence_level, prediction, actual, neighbors):
        # neighbor_mames = [neighbor[0] for neighbor in neighbors]
        row = {"iteration": itaretion, "graph_name": graph_name,  "user_guid": user_guid,  "decision_model": decision_model, "confidence_level": confidence_level, "prediction": prediction, "actual": actual,"neighbors":neighbors}
        return row

    def _create_results_dictionary(self):

        results_dictionary_compenents = []
        for list_name in self._order_of_results_dictionary:
            elements = getattr(self, "_"+ list_name)
            results_dictionary_compenents.append(elements)

        self._result_container = ResultsContainer(self._path, self._results_table_file_name, self._column_names_for_results_table, results_dictionary_compenents)
        self._results_dict = self._result_container.get_results()
        return self._results_dict


    def _predict_on_unlabeled_authors_and_save(self, selected_combination, labeled_author_dict, unlabeled_author_guid_index_field_dict):
        targeted_field_name_as_class = selected_combination[0][0]
        similarity_function_name = selected_combination[0][1]
        selected_k = selected_combination[0][2]
        selected_decision_model = selected_combination[0][3]
        similarity_func_connections = self._db.get_author_connections_by_connection_type(similarity_function_name)

        selected_connections = pd.DataFrame(similarity_func_connections, columns=['source_id', 'destination_id', 'weight'])

        labeled_author_tuples = [(author_guid, targeted_class) for author_guid, targeted_class in labeled_author_dict.items()]

        labeled_authors_df = pd.DataFrame(labeled_author_tuples, columns=['id', 'class'])
        labeled_authors_index = labeled_authors_df.set_index(keys=['id']).index
        selected_connections_source_index = selected_connections.set_index(keys=['source_id']).index
        selected_connections_dest_index = selected_connections.set_index(keys=['destination_id']).index
        source_unlabeled_dest_labeled_connections = selected_connections[(~selected_connections_source_index.isin(labeled_authors_index) & selected_connections_dest_index.isin(labeled_authors_index))]  # source is test, and dest is train
        source_labeled_dest_unlabeled_connections = selected_connections[(selected_connections_source_index.isin(labeled_authors_index) & ~selected_connections_dest_index.isin(labeled_authors_index))]  # opposite from above

        unlabeled_labeled_connections = source_unlabeled_dest_labeled_connections.append(source_labeled_dest_unlabeled_connections)

        unlabeled_author_guids = list(unlabeled_author_guid_index_field_dict.keys())
        unlabeled_index_field = list(unlabeled_author_guid_index_field_dict.values())

        unlabeled_predictions_dataframe = pd.DataFrame(unlabeled_index_field, columns=[self._index_field_for_predictions])

        confidence_levels = []
        predictions = []

        for unlabeled_author_guid in unlabeled_author_guids:
            neighbors = self.get_neighbors(unlabeled_author_guid, labeled_authors_df, unlabeled_labeled_connections, selected_k)

            confidence_level, prediction = getattr(self, selected_decision_model)(neighbors)
            confidence_levels.append(confidence_level)
            predictions.append(prediction)

        predicted_dataframe = pd.DataFrame(predictions,
                                                       columns=["predicted"])

        for num, targeted_class in self._num_to_targeted_class_dict.items():
            predicted_dataframe = predicted_dataframe.replace(to_replace=num, value=targeted_class)

        prediction_dataframe = pd.DataFrame(confidence_levels,
                                                       columns=["prediction"])

        unlabeled_authors_dataframe = pd.concat([unlabeled_predictions_dataframe, predicted_dataframe, prediction_dataframe], axis=1)


        full_path = self._path + "predictions_on_unlabeled_authors_" + targeted_field_name_as_class + "_" + \
                    similarity_function_name + "_" + str(selected_k) + "_" + selected_decision_model +  "_features.csv"
        unlabeled_authors_dataframe.to_csv(full_path, index=False)

        table_name = "unlabeled_predictions"
        self._db.drop_unlabeled_predictions(table_name)

        engine = self._db.engine
        unlabeled_authors_dataframe.to_sql(name=table_name, con=engine)
