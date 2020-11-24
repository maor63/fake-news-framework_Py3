from sklearn.model_selection import train_test_split
import copy
from commons.consts import DistancesFromTargetedClass, Author_Type, Author_Subtype
from configuration.config_class import getConfig
import pandas as pd
from commons.commons import *
from .graph_helper import GraphHelper
import networkx as nx
import math
from DB.schema_definition import AuthorFeatures

class DistancesFromTargetedClassFeatureGenerator:
    def __init__(self, db, **kwargs):
        self._db = db
        self._targeted_classes = getConfig().eval("DEFAULT","targeted_classes")
        if 'authors' in kwargs:
            self._authors = kwargs['authors']
            self._author_dict = create_author_dictionary(self,self._authors)
            self._author_distances, self._author_types_and_sub_types_in_dictionary = self.create_authors_distances_dict()


        self._author_types = [Author_Type.GOOD_ACTOR, Author_Type.BAD_ACTOR]
        self._author_sub_types = [Author_Subtype.PRIVATE, Author_Subtype.COMPANY, Author_Subtype.NEWS_FEED,
                                  Author_Subtype.SPAMMER, Author_Subtype.BOT, Author_Subtype.CROWDTURFER,
                                  Author_Subtype.ACQUIRED]

        start_date = getConfig().get("DEFAULT", "start_date").strip("date('')")
        self._window_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        window_size = datetime.timedelta(seconds=int(getConfig().get("DEFAULT", "window_analyze_size_in_sec")))
        self._window_end = self._window_start + window_size
        self._train_size = getConfig().eval(self.__class__.__name__, DistancesFromTargetedClass.TRAIN_SIZE)
        self._distances_statistics = getConfig().eval(self.__class__.__name__, DistancesFromTargetedClass.DISTANCES_STATS)
        self._calculate_distances_for_unlabeled = getConfig().eval(self.__class__.__name__, DistancesFromTargetedClass.CALCULATE_DISTANCES_FOR_UNLABELED)

        self._graphs = kwargs["graphs"]
        self._graph_directed = kwargs['graph_directed']
        self._graph_types = kwargs['graph_types']
        self._graph_weights = kwargs['graph_weights']
        self.load_graphs()

    def load_graphs(self):
        graph_helper = GraphHelper(self._db)
        self._graphs = graph_helper.load_graphs(self._graphs, self._graph_types, self._graph_directed,
                                                self._graph_weights, self._author_dict)

    def execute(self):
        info_msg = "executing " + self.__class__.__name__
        logging.info(info_msg)
        total_authors = len(self._authors)
        authors_features = []

        for graph_name, graph in list(self._graphs.items()):
            info_msg = "processing " + graph_name + " graph "
            logging.info(info_msg)
            # authors_to_remove include authors used as training set for measuring distances,
            # All the features related to these authors should be deleted
            authors_to_remove = []
            all_authors_dataframe = pd.DataFrame(list(self._author_dict.values()),
                                                 columns=["author_osn_id", "author_type", "author_sub_type",
                                                          "author_guid"])
            #TODO replace author_type and sub_type for a single parameter from config file.
            #TODO write in config.ini two index_fields: index_field_author_connections = author_osn_id
            #TODO and index_field_author_features = author_guid. Two use as column instead of hardcoded columns
            # look example in experimental_environment

            #consider only authors that appear on the graph
            all_authors_dataframe = all_authors_dataframe[all_authors_dataframe['author_osn_id'].isin(graph.nodes())]
            authors_to_remove = self.calculate_distances(all_authors_dataframe, authors_features, graph,
                                                         labeled=True, train_size=self._train_size)
            if self._calculate_distances_for_unlabeled:
                self.calculate_distances(all_authors_dataframe, authors_features, graph)

            if authors_to_remove:
                self._authors = [x for x in self._authors if x.author_guid not in authors_to_remove]
                authors_features = [x for x in authors_features if x.author_guid not in authors_to_remove]

            for author in authors_features:
                self._db.update_author_features(author)
            self._db.commit()

        info_msg = "finished executing " + self.__class__.__name__+" , "+str(total_authors)+" authors processed "
        logging.info(info_msg)

    def retrieve_author_by_type(self, data_frame, author_type=None, labeled=None, unlabeled=None):
        #TODO unifiy these function and the _retreive_unlabeled_authors_dataframe from experimental environment
        #TODO refactor this and move into a single function in commons, and verify that it is used in the experimental environment
        authors_data_frame = None
        if author_type in self._author_sub_types:
            authors_data_frame = data_frame[data_frame.author_sub_type == author_type]
        elif author_type in self._author_types:
            authors_data_frame = data_frame[data_frame.author_type == author_type]
        elif labeled:
            authors_data_frame = data_frame[data_frame.author_type.notnull()]
        elif unlabeled:
            authors_data_frame = data_frame[(data_frame['author_type'].isnull())]
        return authors_data_frame

    def create_authors_distances_dict(self):
        #TODO use dict comprehension to create sub_type_path, use Enum instead of author_types_sub_types_in_dict..

        author_guid_path_to_labeled_dict = {}
        author_sub_type_paths_dict = {
            "private": pd.Series(),
            "company": pd.Series(),
            "news_feed": pd.Series(),
            "spammer": pd.Series(),
            "bot": pd.Series(),
            "bad_actor": pd.Series(),
            "good_actor": pd.Series()
        }

        author_types_and_sub_types_in_dictionary = list(author_sub_type_paths_dict.keys())

        for author in self._authors:
            author_guid = author.author_guid
            author_sub_type_series_dict = copy.deepcopy(author_sub_type_paths_dict)
            author_guid_path_to_labeled_dict[author_guid] = author_sub_type_series_dict

        return author_guid_path_to_labeled_dict, author_types_and_sub_types_in_dictionary


    def calculate_distances_by_author_type(self, train_set_dataframe, test_set_dataframe, graph):
        for index, test_record in test_set_dataframe.iterrows():
            for author_type_or_sub_type in self._author_types_and_sub_types_in_dictionary: #TODO change this to target class only
                author_type_or_sub_type_data_frame = self.retrieve_author_by_type(train_set_dataframe, author_type=author_type_or_sub_type)
                authors = author_type_or_sub_type_data_frame.values.tolist()
                if len(authors) > 0:
                    for author in authors:
                        if nx.has_path(graph, test_record["author_osn_id"], author[0]):
                            shortest_path_length = nx.shortest_path_length(graph, test_record["author_osn_id"], author[0])
                            # in order to append item to series we have to convert it to series prior appending
                            shortest_path_length = pd.Series([float(shortest_path_length)])

                            #append on series return new series. Because of that we save the append to our dictionary
                            self._author_distances[test_record.author_osn_id][author_type_or_sub_type] = \
                                self._author_distances[test_record.author_osn_id][author_type_or_sub_type].append(shortest_path_length)

    def create_features_from_calculated_distances(self, authors_features):
        # key             #value
        for author_guid, author_sub_type_paths_dict in list(self._author_distances.items()):
            author = self._author_dict[author_guid]
            for author_sub_type in self._author_types_and_sub_types_in_dictionary:
                series = author_sub_type_paths_dict[author_sub_type]

                if DistancesFromTargetedClass.MEAN in self._distances_statistics:
                    mean = series.mean()
                    if math.isnan(mean):
                        mean = None
                    #TODO add parameter attenuation function, that should be applied to the distance
                    #TODO think of several types of attenuation functions, exponentially decreasing, logarithmically decreasing, etc..
                    mean_dist_to_author_sub_type_feature = AuthorFeatures(author[3],self._window_start,
                                                                                       self._window_end,
                                                                                       str("mean_dist_to_" + author_sub_type),
                                                                                       mean)
                    authors_features.append(mean_dist_to_author_sub_type_feature)

                if DistancesFromTargetedClass.MIN in self._distances_statistics:
                    min_in_series = series.min()
                    if math.isnan(min_in_series):
                        min_in_series = None

                    min_dist_to_author_sub_type_feature = AuthorFeatures(author[3], self._window_start,
                                                                         self._window_end,
                                                                         str("min_dist_to_" + author_sub_type),
                                                                         min_in_series)
                    authors_features.append(min_dist_to_author_sub_type_feature)

    #TODO ASK Rami if it is necesary to calculate features for unlabeled authors!!!!
    #TODO add threshold for unlabeled and calculate the features of unlabeled all the time.

    def calculate_distances(self, authors_dataframe, authors_features, graph, labeled=False, train_size=None):
        if len(graph.nodes()) == 0:
            return
        labeled_authors_dataframe = self.retrieve_author_by_type(authors_dataframe, labeled=True)
        labeled_authors_dataframe.reset_index(drop=True, inplace=True)

        if labeled:
            train_dataframe, test_set_dataframe = train_test_split(labeled_authors_dataframe,
                                                                                train_size=train_size)
        else:
            unlabeled_authors_dataframe = self.retrieve_author_by_type(authors_dataframe, unlabeled=True)
            train_dataframe = labeled_authors_dataframe
            test_set_dataframe = unlabeled_authors_dataframe
            test_set_dataframe.reset_index(drop=True, inplace=True)

        self.calculate_distances_by_author_type(train_dataframe, test_set_dataframe, graph)
        self.create_features_from_calculated_distances(authors_features)

        if labeled:
            author_guid_train_series = train_dataframe.author_guid
            return author_guid_train_series.tolist()