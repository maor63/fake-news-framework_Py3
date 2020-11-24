# Created by jorgeaug at 30/06/2016

import datetime
import logging
import time
import timeit
import numpy as np
import pandas as pd

from DB.schema_definition import AuthorFeatures, Author
from configuration.config_class import getConfig
from dataset_builder.feature_extractor.feature_argument_parser import ArgumentParser
from collections import defaultdict


class BaseFeatureGenerator(ArgumentParser):
    def __init__(self, db, **kwargs):

        super(BaseFeatureGenerator, self).__init__(db)
        self._load_limit = self._config_parser.eval("FeatureExtractor", "load_limit")
        self._max_objects_save = self._config_parser.eval("FeatureExtractor", "max_objects_save")
        # self._features_names_count = 0
        self._db = db
        self._config_parser = getConfig()
        self._targeted_social_network = self._config_parser.get("DEFAULT", "social_network_name")

        start_date = self._config_parser.get("DEFAULT", "start_date").strip("date('')")
        self._window_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self._window_size = datetime.timedelta(
            seconds=int(self._config_parser.get("DEFAULT", "window_analyze_size_in_sec")))
        # self._window_end = self._window_start + self._window_size
        self._domain = str(self._config_parser.get(self.__class__.__name__, "domain"))

        if 'authors' in kwargs and 'posts' in kwargs:
            self.authors = kwargs['authors']
            self.author_guid_posts_dict = kwargs['posts']
        else:
            raise Exception('Author object was not passed as parameter')

        if 'measure' in kwargs:
            self._measure = kwargs['measure']
        if 'calculator_type' in kwargs:
            self._calculator_type = kwargs['calculator_type']
        if 'aggregation_function' in kwargs:
            self._aggregation_function = kwargs['aggregation_function']
        if 'graph_type' in kwargs:
            self._graph_type = kwargs['graph_type']
        if 'targeted_class_field_name' in kwargs:
            self._targeted_class_field_name = kwargs['targeted_class_field_name']

    def cleanUp(self, window_start=None):
        pass

    def execute(self, window_start=None):
        start_time = time.time()
        info_msg = "execute started for " + self.__class__.__name__
        logging.info(info_msg)

        total_authors = len(self.authors)
        processed_authors = 0
        features = getConfig().eval(self.__class__.__name__, "feature_list")

        authors_features = []
        counter = 0
        for author in self.authors:
            counter += 1
            author_guid = author.author_guid
            if author_guid in list(self.author_guid_posts_dict.keys()):
                posts = self.author_guid_posts_dict[str(author.author_guid)]
                getattr(self, 'cleanUp')()
                kwargs = {'author': author, 'posts': posts}
                author_guid = author.author_guid
                for feature in features:
                    author_feature = self.run_and_create_author_feature(kwargs, author_guid, feature)
                    authors_features.append(author_feature)

            processed_authors += 1
            if processed_authors % 100 == 0:
                print("\r processed authors " + str(processed_authors) + " from " + str(total_authors), end="")

            if len(authors_features) > self._max_objects_save:
                self.submit_author_features_to_db(authors_features)
                authors_features = []

        if authors_features:
            self.submit_author_features_to_db(authors_features)

        end_time = time.time()
        diff_time = end_time - start_time
        print('execute finished in ' + str(diff_time) + ' seconds')

    def is_well_defined(self):
        return True

    def submit_author_features_to_db(self, authors_features):  # TODO
        print('\n Beginning merging author_features objects')
        counter = 0
        for author_features_row in authors_features:
            counter += 1
            self._db.update_author_features(author_features_row)
            if counter == 1000000:
                print("\r " + str(self.__class__.__name__) + " merging author-features objects", end="")
                self._db.commit()
                counter = 0
        if counter != 0:
            self._db.commit()
        print('Finished merging author_features objects')

    def run_and_create_author_feature(self, kwargs, id_val, feature, aggregated_functions=None):
        try:
            result = getattr(self, feature)(**kwargs)
            if aggregated_functions:
                result = [x for x in result if x != -1]
                if result:
                    return [self._create_feature(feature, id_val, func(result), '_{}'.format(func.__name__)) for func
                            in aggregated_functions]
                else:
                    return [self._create_feature(feature, id_val, -1, '_{}'.format(func.__name__)) for func in
                            aggregated_functions]
            else:
                author_feature = self._create_feature(feature, id_val, result)
            return author_feature
        except Exception as e:
            info_msg = e
            logging.error(info_msg)

    def _create_feature(self, feature, id_val, result, suffix=''):
        author_feature = AuthorFeatures()
        author_feature.author_guid = id_val
        author_feature.window_start = self._window_start
        author_feature.window_end = self._window_end
        subclass_name = self.__class__.__name__
        author_feature.attribute_name = '{}_{}{}'.format(subclass_name, feature, suffix)
        author_feature.attribute_value = str(result)
        return author_feature

    def run_and_create_author_feature_with_given_value(self, author_guid, value, feature_name):
        try:
            result = value
            author_feature = self._create_feature(feature_name, author_guid, result)
            return author_feature
        except Exception as e:
            info_msg = e
            print(info_msg)
            logging.error(info_msg + str(value))

    def convert_posts_to_dataframe(self, posts):
        '''
          Input: list of posts
          Output: DataFrame with two columns: date and content where each row represents a post
        '''
        return pd.DataFrame([[post.date, post.content] for post in posts], columns=['date', 'content'])

    def insert_author_features_to_db(self, authors_features):
        logging.info("Inserting authors features to db")
        start_time = time.time()
        if authors_features:
            self.submit_author_features_to_db(authors_features)

        end_time = time.time()
        diff_time = end_time - start_time
        print('execute finished in ' + str(diff_time) + ' seconds')

    def load_target_field_for_id(self, targeted_fields):
        author_id_texts_dict = {}
        for targeted_fields_dict in targeted_fields:
            table_name = targeted_fields_dict["table_name"]
            id_field = targeted_fields_dict["id_field"]
            targeted_field_name = targeted_fields_dict["targeted_field_name"]
            key_tuple = table_name + "-" + id_field + "-" + targeted_field_name
            where_clauses = targeted_fields_dict["where_clauses"]
            author_id_texts_dict[key_tuple] = {}

            tuples = self._db.get_records_by_id_targeted_field_and_table_name(id_field, targeted_field_name,
                                                                              table_name, where_clauses)
            i = 1
            for tuple in tuples:
                i += 1
                author_id = tuple[0]
                # text could be content, description, etc
                text = tuple[1]
                if author_id not in author_id_texts_dict[key_tuple]:
                    author_id_texts_dict[key_tuple][author_id] = []
                    author_id_texts_dict[key_tuple][author_id].append(text)
                else:
                    author_id_texts_dict[key_tuple][author_id].append(text)
            if i == 1:
                author_id_texts_dict[key_tuple] = {}
        return author_id_texts_dict

    def get_key_tuple(self, targeted_fields_dict):
        table_name = targeted_fields_dict["table_name"]
        id_field = targeted_fields_dict["id_field"]
        targeted_field_name = targeted_fields_dict["targeted_field_name"]
        key_tuple = table_name + "-" + id_field + "-" + targeted_field_name
        return key_tuple

    @staticmethod
    def create_author_feature(feature_name, author_guid, attribute_value, window_start, window_end):
        feature = AuthorFeatures()
        feature.author_guid = author_guid
        feature.attribute_name = feature_name
        feature.attribute_value = attribute_value
        feature.window_start = window_start
        feature.window_end = window_end
        return feature

    def _get_destination_tuples(self, args):
        destination_table_name = args['destination']['table_name']
        destination_text_field_name = args['destination']['target_field']
        destination_id = args['destination']['id']
        dest_where_clauses = {}
        if 'where_clauses' in args['destination']:
            dest_where_clauses = args['destination']['where_clauses']
        dest_id_target_field_tuples = self._db.get_records_by_id_targeted_field_and_table_name(destination_id,
                                                                                               destination_text_field_name,
                                                                                               destination_table_name,
                                                                                               dest_where_clauses)
        return dest_id_target_field_tuples

    def _calc_author_features_from_source_id_targets(self, source_targets_dict_item, source_id_source_element_dict,
                                                     targeted_fields_dict, aggregated_functions=None):
        self._features = self._config_parser.eval(self.__class__.__name__, "feature_list")
        author_features = []
        source_id, destination_target_elements = source_targets_dict_item
        author = source_id_source_element_dict[source_id]
        kwargs = self._get_feature_kwargs(source_targets_dict_item, author, targeted_fields_dict)
        times = []
        for feature in self._features:
            start = timeit.default_timer()
            if aggregated_functions:
                author_feature_list = self.run_and_create_author_feature(kwargs, source_id, feature,
                                                                         aggregated_functions)
                if author_feature_list:
                    author_features.extend(author_feature_list)
            else:
                author_features.append(self.run_and_create_author_feature(kwargs, source_id, feature))
            end = timeit.default_timer()
            times.append(end - start)

        return author_features, times

    def _get_feature_kwargs(self, source_targets_dict_item, author, targeted_fields_dict):
        source_id, destination_target_elements = source_targets_dict_item
        if "destination" in targeted_fields_dict and targeted_fields_dict["destination"] != {}:
            table_name = targeted_fields_dict['destination']['table_name']
            target_field_name = targeted_fields_dict['destination']['target_field']
        else:
            target_field_name = targeted_fields_dict['source']['target_field']
            table_name = targeted_fields_dict['source']['table_name']
        kwargs = {table_name: destination_target_elements}
        kwargs['target'] = [getattr(element, target_field_name) for element in destination_target_elements]
        kwargs['author'] = author
        if 'posts' not in kwargs and isinstance(author, Author):
            kwargs['posts'] = self._db.get_posts_by_author_guid(author.author_guid)
        return kwargs

    def get_source_id_destination_target_field(self, targeted_fields_dict):
        # source_id_target_items = self._get_source_id_target_elements(targeted_fields_dict)
        for source_id_target_items in self._get_source_id_target_elements_generator(targeted_fields_dict,
                                                                                    self._load_limit):
            print()
            if "destination" in targeted_fields_dict and targeted_fields_dict["destination"] != {}:
                target_field_name = targeted_fields_dict['destination']['target_field']
            else:
                target_field_name = targeted_fields_dict['source']['target_field']
            # source_ids = source_id_target_items.keys()
            # source_id_with_features = set(self._db.get_authors_with_features(self.__class__.__name__, source_ids))

            source_id_target_fields = defaultdict()
            for source_id, elements in source_id_target_items.items():
                # if source_id not in source_id_with_features:
                source_id_target_fields[source_id] = [getattr(element, target_field_name) for element in elements]
            yield source_id_target_fields

    def _get_author_features_using_args(self, targeted_fields, suffix=''):
        features = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._features_names_count = len(features)
        for targeted_fields_dict in targeted_fields:
            authors_features = self._get_author_features_using_arg(targeted_fields_dict, suffix)
            # self.insert_author_features_to_db(authors_features)
            self._db.add_author_features_fast(authors_features)
        self._db.session.commit()

    def _get_author_features_using_arg(self, targeted_fields_dict, suffix=''):
        authors_features = []
        for source_id_element_dict, source_targets_dict in self._load_data_using_arg(targeted_fields_dict):
            authors_features += self._get_features(source_id_element_dict, source_targets_dict,
                                                   suffix, targeted_fields_dict)
        authors_features = self._add_suffix_to_author_features(authors_features, suffix)
        return authors_features

    def _add_suffix_to_author_features(self, authors_features, suffix):
        authors_features = [feature for feature in authors_features if feature is not None]
        if suffix != '':
            for authors_feature in authors_features:
                authors_feature.attribute_name += '_{}'.format(suffix)
        return authors_features

    def _load_data_using_arg(self, targeted_fields_dict):
        print("Get sourse id target dict")
        source_id_target_elements_dict_gen = self._get_source_id_target_elements_generator(targeted_fields_dict,
                                                                                           self._load_limit)
        total_sources = 0
        for source_id_target_elements_dict in source_id_target_elements_dict_gen:
            source_ids = list(source_id_target_elements_dict.keys())
            total_sources += len(source_ids)
            print("Get sourse id source element dict. {} sources".format(total_sources))
            source_id_source_element_dict = self._get_source_id_source_element_dict(source_ids, targeted_fields_dict)
            yield source_id_source_element_dict, source_id_target_elements_dict

    def _get_features(self, source_id_source_element_dict, source_id_target_elements_dict, suffix,
                      targeted_fields_dict, aggregated_functions=None):
        source_count = len(source_id_target_elements_dict)
        authors_features = []
        i = 1
        run_times = []
        for source_id_target_field_dict_item in source_id_target_elements_dict.items():
            msg = "\rextract author features {}/{}".format(i, source_count)
            print(msg, end="")
            i += 1
            features, times = self._calc_author_features_from_source_id_targets(source_id_target_field_dict_item,
                                                                                source_id_source_element_dict,
                                                                                targeted_fields_dict,
                                                                                aggregated_functions)
            # self._add_suffix_to_author_features(features, suffix)
            authors_features.extend(features)
            run_times.append(times)

            # if len(authors_features) > self._max_objects_save:
            #     print()
            #     self._db.add_author_features_fast(authors_features)
            #     authors_features = []
        print()
        if run_times:
            times_df = pd.DataFrame(np.array(run_times), columns=self._features)
            print(times_df.sum())

        return authors_features
