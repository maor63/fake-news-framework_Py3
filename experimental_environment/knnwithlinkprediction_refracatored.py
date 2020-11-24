from sqlalchemy import Column, Unicode, Integer

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

class KNNWithLinkPrediction_Refactored(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._k = self._config_parser.eval(self.__class__.__name__, "k")
        self._index_field_for_predictions = self._config_parser.get(self.__class__.__name__, "index_field_for_predictions")
        self._connection_types = self._config_parser.get(self.__class__.__name__, "connection_type")
        self._desicion_models = self._config_parser.eval(self.__class__.__name__, "decision_models")


        self._connections = []
        self._decision_by_author_to_config = {} # test config = K to desicion function to

    def set_up(self):
        pass

    def execute(self, window_start=None):
        self._labeled_author_dict, unlabeled_author_dict, unlabeled_author_guid_index_field_dict = self._db.create_author_dictionaries(self._index_field_for_predictions, self._domain)
        connections = self._db.get_author_connections_by_connection_type(self._connection_types)
        self._connections = self._parse_connections(connections)
        for k in self._k:
            for unlabled_author in unlabeled_author_dict:
                labled_neighbors_links = self.get_authors_links_to_labled_authors(unlabled_author)
                k_nearest_links = self.choose_k_nearest_from_neighbours_link(labled_neighbors_links, k)
                for desicion_model in self._desicion_models:

                    authors_label = self.get_label_from_links_by_decision_models(unlabled_author, k_nearest_links, desicion_model)
                # self.update_authors_label(unlabled_author, authors_label)

    def get_connections_to_labeled_authors_where_author_is_source(self, author_guid):
        ans = []
        for connection in self._connections:
            if connection.source == author_guid:
                ans.append(connection)
        return ans

    def get_connections_to_labeled_authors_where_author_is_destination(self, author_guid):
        ans = []
        for connection in self._connections:
            if connection.destination == author_guid:
                ans.append(connection)
        return ans

    def get_authors_links_to_labled_authors(self, author):
        ans = self.get_connections_to_labeled_authors_where_author_is_source(author) + self.get_connections_to_labeled_authors_where_author_is_destination(author)
        return ans

    def choose_k_nearest_from_neighbours_link(self, links, k):
        links.sort(key=lambda link: link.weight)
        return links[0:k]

    def get_label_from_links_by_decision_models(self, target_author_guid, links, desicion_model):
        return getattr(self, desicion_model)(target_author_guid, links)

    def majority_voting(self, target_author_guid, links):
        votes_dict = {}
        for link in links:
            label = self._get_link_label(link, target_author_guid)
            self._raise_label_value(votes_dict,label,1)
        return self._get_max_from_dict(votes_dict)

    def weighted_majority_voting(self, target_author_guid, links):
        votes_dict = {}
        for link in links:
            label = self._get_link_label(link, target_author_guid)
            self._raise_label_value(votes_dict,label,link.weight)
        return self._get_max_from_dict(votes_dict)

    def _get_max_from_dict(self, dict):
        max_key = max(iter(dict.items()), key=operator.itemgetter(1))[0]
        return  max_key

    def _get_link_label(self, link, un_labeled):
        if link.destination == un_labeled:
            return self._labeled_author_dict[link.source]
        if link.source == un_labeled:
            return self._labeled_author_dict[link.destination]
        else:
            raise Exception("Bad author guid")
    def _raise_label_value(self, votes_dict, label, weight):
        if label not in list(votes_dict.keys()):
            votes_dict[label]=0
        votes_dict[label] += 1
        return

    def update_authors_label(self, author_guid, label):
        pass

    def _parse_connections(self, connections):
        ans = []
        for connection in connections:
            _source = connection[0]
            _destinetion = connection[1]
            _weight = connection[2]
            link = Link(_source,_destinetion,_weight)
            ans.append(link)
        return ans

class Link():
    source = Column(Unicode, default=None)
    destination = Column(Unicode, default=None)
    weight = Column(Integer, default=0)

    def __init__(self, _source, _destination, _weight):
        self.source =_source
        self.destination = _destination
        self.weight = _weight

    def __repr__(self):
        return "<Link(source='%s', destination='%s', weight='%i')> " % (
            self.source,self.destination,self.weight)
