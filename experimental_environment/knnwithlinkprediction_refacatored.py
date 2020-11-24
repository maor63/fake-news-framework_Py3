import csv
import math
import operator
import random

import itertools
from collections import OrderedDict

import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, accuracy_score
from sqlalchemy import Column, Unicode, Float

from preprocessing_tools.abstract_controller import AbstractController


class KNNWithLinkPrediction_Refactored(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._k = self._config_parser.eval(self.__class__.__name__, "k")
        self._num_of_iterations = int(self._config_parser.get(self.__class__.__name__, "num_of_iterations"))
        self._index_field_for_predictions = self._config_parser.get(self.__class__.__name__, "index_field_for_predictions")
        self._connection_types = self._config_parser.get(self.__class__.__name__, "connection_type")
        self._desicion_models = self._config_parser.eval(self.__class__.__name__, "decision_models")
        self._result_authors_file_name = self._config_parser.get(self.__class__.__name__, "result_author_file_name")
        self._result_classifier_file_name = self._config_parser.get(self.__class__.__name__, "result_classifier_file_name")
        self._train_percent = int(self._config_parser.get(self.__class__.__name__,"train_percent"))
        self._randomly_classify_unlabelled = self._config_parser.eval(self.__class__.__name__, "randomly_clasify_unlabled")
        self._classification_options = self._config_parser.eval(self.__class__.__name__, "clasification_options")
        self._targeted_class = self._config_parser.get(self.__class__.__name__, "targeted_class")
        self._similarity_function = self._config_parser.get(self.__class__.__name__, "similarity_function")
        self._update_targeted_class_in_db = self._config_parser.eval(self.__class__.__name__, "update_targeted_class_in_db")
        self._random_seed =int(self._config_parser.get(self.__class__.__name__,"random_seed"))
        self._connections = []

        self._result_dataframe_of_author = None
        self._result_dataframe_of_classifier = None
        self._data_holder = None
    def set_up(self):
        pass

    def execute(self, window_start=None):
        self.load_and_parse_data()

        for iteration in range(1, self._num_of_iterations):
            for k, desicion_model in itertools.product(self._k, self._desicion_models):
                predicted_labels = []
                actual_labels = []
                correctly_classified = 0
                for author in self._data_holder.test_set:
                    authors_label, result = self.get_result_of_prediction(author, desicion_model, iteration, k)
                    authors_actual_label= self._get_label_by_author(author)
                    print("labeled: "+author+" as "+authors_label+" actual lable: "+authors_actual_label)
                    if authors_label == authors_actual_label:
                        correctly_classified+=1
                    predicted_labels.append(self._classification_options[authors_label])
                    actual_labels.append(self._classification_options[authors_actual_label])
                    self.update_authors_label(result)

                self.measure_and_save_classifier_performance(actual_labels, correctly_classified, desicion_model,
                                                             iteration, k, predicted_labels)

        # self.save_results_as_csv()

    def load_and_parse_data(self):
        self._original_labeled_author_dict, unlabeled_author_dict, unlabeled_author_guid_index_field_dict = self._db.create_author_dictionaries(
            self._index_field_for_predictions, self._domain)
        self._labled_author_dict = self._original_labeled_author_dict
        connections = self._db.get_author_connections_by_connection_type(self._connection_types)
        self._connections = self._parse_connections(connections)
        self._data_holder = DataHolder(self._original_labeled_author_dict, self._train_percent, self._random_seed,
                                       self._connections)

    def measure_and_save_classifier_performance(self, actual_labels, correctly_classified, desicion_model, iteration, k,
                                                predicted_labels):
        classifier_parameters = (
        self._targeted_class, self._similarity_function, k, desicion_model, iteration, correctly_classified,
        len(self._data_holder.test_set), len(self._data_holder.test_set) - correctly_classified)
        self._calculate_performance(classifier_parameters, predicted_labels, actual_labels)

    def get_result_of_prediction(self, author, desicion_model, iteration, k):
        labeled_neighbors_links = self.get_authors_links_to_labled_authors(author)
        k_nearest_links = self.choose_k_nearest_from_neighbours_link(labeled_neighbors_links, k)
        if len(k_nearest_links) == 0:
            authors_label = self._get_random_lable(self._classification_options, self._random_seed)
            confidence = 0
        else:
            authors_label, confidence = self.get_label_from_links_by_decision_models(author, k_nearest_links,
                                                                                     desicion_model)
        result = Result_of_Author(author, k, desicion_model, self._train_percent, iteration, authors_label,
                                  self._get_label_by_author(author), confidence)
        return authors_label, result

    def save_results_as_csv(self):
        self._result_dataframe_of_author.to_csv(self._result_authors_file_name, encoding='utf-8', index=False)
        self._result_dataframe_of_classifier.to_csv(self._result_classifier_file_name, encoding='utf-8', index=False)

    def get_connections_to_labeled_authors_where_author_is_source(self, author_guid):
        ans = []
        for connection in self._connections:
            if connection.source == author_guid and connection.destination in self._data_holder.anchors:
                ans.append(connection)
        return ans

    def get_connections_to_labeled_authors_where_author_is_destination(self, author_guid):
        ans = []
        for connection in self._connections:
            if connection.destination == author_guid and connection.source in self._data_holder.anchors:
                ans.append(connection)
        return ans

    def get_authors_links_to_labled_authors(self, author):
        ans = self.get_connections_to_labeled_authors_where_author_is_source(author) + self.get_connections_to_labeled_authors_where_author_is_destination(author)
        return ans

    def choose_k_nearest_from_neighbours_link(self, links, k):
        links.sort(key=lambda link: link.weight)
        return links[len(links)-k:len(links)]

    def get_label_from_links_by_decision_models(self, target_author_guid, links, desicion_model):
        return getattr(self, desicion_model)(target_author_guid, links)

    def majority_voting(self, target_author_guid, links):
        votes_dict = {}
        for link in links:
            label = self._get_link_label(link, target_author_guid)
            self._raise_label_value(votes_dict,label,1)

        max_voted_label = self._get_max_from_dict(votes_dict)
        confidence_level = self._get_confidence(votes_dict,max_voted_label)
        return max_voted_label, confidence_level

    def weighted_majority_voting(self, target_author_guid, links):
        votes_dict = {}
        for link in links:
            label = self._get_link_label(link, target_author_guid)
            self._raise_label_value(votes_dict,label,link.weight)
        max_voted_label =  self._get_max_from_dict(votes_dict)
        confidence_level = self._get_confidence(votes_dict,max_voted_label)
        return max_voted_label, confidence_level

    def _get_confidence(self, vote_dict, calculated_max):
        sum_result = sum(vote_dict.values())
        return vote_dict[calculated_max]/float(sum_result)

    def _get_max_from_dict(self, dict):
        max_key = max(iter(dict.items()), key=operator.itemgetter(1))[0]
        return  max_key

    def _get_link_label(self, link, un_labeled):
        if link.destination == un_labeled:
            return self._get_label_by_author(link.source)
        if link.source == un_labeled:
            return self._get_label_by_author(link.destination)
        else:
            raise Exception("Bad author guid")
    def _get_label_by_author(self, author):
        return self._labled_author_dict[author]
    def _raise_label_value(self, votes_dict, label, weight):
        if label not in list(votes_dict.keys()):
            votes_dict[label]=0
        votes_dict[label] += weight
        return

    def update_authors_label(self, result):
        if self._update_author_type_in_db:
            self._db.update_author_type_by_author_guid(result.author_guid,result.label)

        result_list = result.to_list()
        if not os.path.isfile(self._result_authors_file_name) :
            columns = Result_of_Author.columns
            # self._result_dataframe_of_author = pd.DataFrame([result_list], columns=columns)
            # self._result_dataframe_of_author.to_csv(self._result_authors_file_name)
            with open(self._result_authors_file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([columns,result_list])
            return
        # temp_pd = pd.DataFrame([result.to_list()])
        with open(self._result_authors_file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

        # frame_len= len(self._result_dataframe_of_author)
        # self._result_dataframe_of_author.loc[frame_len]= result_list
        return

    def _parse_connections(self, connections):
        ans = []
        for connection in connections:
            _source = connection[0]
            _destination = connection[1]
            _weight = connection[2]
            link = Link(_source,_destination,_weight)
            ans.append(link)
        return ans

    def _calculate_performance(self, classifier_parameter_list, predicted_labels, actual_labels):
        performance_report = precision_recall_fscore_support(actual_labels, predicted_labels, average='binary')
        precision = performance_report[0]
        recall = performance_report[1]
        f1 = performance_report[2]
        accuracy = accuracy_score(actual_labels, predicted_labels)

        try:
            auc = roc_auc_score(actual_labels, predicted_labels)
        except Exception as e:
            auc = -1
        combined_data = classifier_parameter_list + (auc, accuracy, precision, recall, f1)
        print(performance_report)
        if self._result_dataframe_of_classifier is None:
            columns = Result_of_Classification.columns
            self._result_dataframe_of_classifier = pd.DataFrame([combined_data], columns=columns)
            self._result_dataframe_of_classifier.to_csv(self._result_classifier_file_name)
            return
        with open(self._result_classifier_file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(combined_data)
        # fd.write(combined_data)
        # fd.close()
        # self._result_dataframe_of_classifier.loc[len(self._result_dataframe_of_classifier)]= combined_data
        return

    def _update_author_type_in_db(self, author_guid, label):
        author = self._db.get_author_by_guid(author_guid)
        author.author_type = label
        self._db.update_author(author)

    def _get_random_lable(self, classification_option, random_seed):
        random.seed(random_seed)
        random_classification = random.choice(list(classification_option.keys()))
        return random_classification

    # def is_well_defined(self):
    #     try:
    #         attribute_list = ("k", "num_of_iterations", "index_field_for_predictions", "connection_type", "connection_type", "decision_models", "result_author_file_name"
    #                           ,"result_classifier_file_name", "train_percent", "randomly_clasify_unlabled", "clasification_options", "targeted_class", "similarity_function","update_targeted_class_in_db", "random_seed")
    #         self.check_config_has_attributes(attribute_list)
    #         return True
    #     except Exception as e:
    #         raise e
    #         return False
    #     return False

class Result_of_Classification():
    columns = ("targeted_class", "similarity_function","k", "decision_model","iteration", "correctly_classified", "incorrectly_classified", "total", "auc", "accuracy","precision","recall","f1")
    targeted_class = ""
    similarity_function =""
    k = -1
    iteration = -1
    decision_model = ""
    correctly_classified =-1
    incorrectly_classified= -1
    total =-1
    AUC =-1
    accuracy=-1
    precision=-1
    recall=-1
    f1 = -1


class Result_of_Author():
    columns = ('author_guid', 'k', 'desicion_model','train_percent','iteration', 'label','actual_label','confidence')
    author_guid = '-1'
    k = -1
    desicion_model = 'None'
    train_percent = 0
    iteration = 0
    label = 'None'
    actual = 'None'
    confidence = 'None'
    def __init__(self, author_guid, k, desicion_model, train_percent, iteration, label, actual, confidence_level):
        self.author_guid=author_guid
        self.k=k
        self.desicion_model=desicion_model
        self.iteration = iteration
        self.train_percent = train_percent
        self.label=label
        self.actual = actual
        self.confidence_level = confidence_level

    def to_list(self):
        return (self.author_guid,self.k,self.desicion_model,self.train_percent,self.iteration,self.label,self.actual,self.confidence_level)
    def to_dict(self):
        return {'author_guid':self.author_guid, 'k':self.k,'desicion_model':self.desicion_model,'train_percent':self.train_percent, 'iteration':self.iteration,'label':self.label,'actual':self.actual,'confidence_level':self.confidence_level}

class Link():
    source = Column(Unicode, default=None)
    destination = Column(Unicode, default=None)
    weight = Column(Float, default=0.0)

    def __init__(self, _source, _destination, _weight):
        self.source =_source
        self.destination = _destination
        self.weight = _weight

    def __repr__(self):
        return "<Link(source='%s', destination='%s', weight='%f')> " % (
            self.source,self.destination,self.weight)

class DataHolder():
    anchors = []
    test_set = []
    connections = []
    def __init__(self, labbled_authors, train_percent, random_seed, connections):
        self._connections = connections
        labeled_authors = list(labbled_authors)
        n = math.floor((1-(float(train_percent)/100)) * len(labeled_authors))
        num_to_choose = int(n)
        random.seed(random_seed)
        random.shuffle(labeled_authors)
        self.anchors = labeled_authors[0:num_to_choose]
        self.test_set = labeled_authors[num_to_choose:len(labeled_authors)]

    def randomly_classify_authors_to_labelled(self, un_labelled_authors, labelled_author_dict, classification_option, random_seed):
        for author in un_labelled_authors:
            random.seed(random_seed)
            random_classification = random.choice(classification_option)
            labelled_author_dict[author]= random_classification
            self.anchors.append(author)