
from nltk import ngrams
from nltk.corpus import stopwords
import re
import pandas as pd
import string
from nltk.stem.porter import *
import nltk
from DB.schema_definition import AuthorFeatures
from commons.commons import *
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.abstract_controller import AbstractController


class N_Grams_Feature_Generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        nltk.download('stopwords')
        self._n = self._config_parser.eval(self.__class__.__name__, "n")
        self._maximal_records_without_saving = self._config_parser.eval(self.__class__.__name__,
                                                                        "maximal_records_without_saving")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._remove_stopwords = self._config_parser.eval(self.__class__.__name__, "remove_stopwords")
        self._stemming = self._config_parser.eval(self.__class__.__name__, "stemming")
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        # self._posts_by_author_guid = self._db.get_author_guid_to_posts_dict()
        self._threshold = self._config_parser.eval(self.__class__.__name__, "threshold")
        self._create_features = self._config_parser.eval(self.__class__.__name__, "create_features")

    def execute(self, window_start=None):
        for target_field in self._targeted_fields:
            self._posts_by_author_guid = self._get_source_id_target_elements(target_field)
            author_features = []
            author_feature_counter = 0
            for k, current_n in enumerate(self._n):
                print('Generating features for {} n-grams {}/{}'.format(current_n, str(k + 1), len(self._n)))
                self._general_n_gram_count_dict = {}
                i = 0
                # for trump_tweet in self._trump_tweets:
                for j, author_guid in enumerate(self._posts_by_author_guid.keys()):
                    self._n_gram_count_dict = {}
                    for record in self._posts_by_author_guid[author_guid]:
                        i += 1
                        words = self.extract_and_clean_content(record)
                        n_grams = ngrams(words, current_n)
                        for n_gram in n_grams:
                            key = ""
                            for gram in n_gram:
                                key = key + gram + "_"
                            key = key[:-1]
                            key = key.lower()
                            self._update_dictionary(key, self._n_gram_count_dict)
                            self._update_dictionary(key, self._general_n_gram_count_dict)
                        if self._create_features:
                            for key, count in self._n_gram_count_dict.items():
                                if count < self._threshold:
                                    continue
                                feature_name = str(current_n) + "_gram_" + key

                                author_feature = self.create_author_feature(feature_name, author_guid, count,
                                                                            self._window_start, self._window_end)
                                author_feature_counter += 1
                                author_features.append(author_feature)
                                print('\rgenerate for author {}/{} n-grams features {}'.format(j, len(
                                    list(self._posts_by_author_guid.keys())), len(author_features)), end='')

                                if author_feature_counter > self._maximal_records_without_saving:
                                    self._db.add_author_features_fast(author_features)
                                    author_feature_counter = 0
                                    author_features = []

                    keys = list(self._general_n_gram_count_dict.keys())
                    values = list(self._general_n_gram_count_dict.values())

                    length = len(keys)
                    n_gram_size_array = [current_n] * length
                    dict = {"n_gram_size": n_gram_size_array,
                            "words": keys,
                            "count": values}
                    df = pd.DataFrame(dict, columns=['n_gram_size', 'words', 'count'])
                    df = df.sort_values(by=['count'], ascending=False)
                    df.to_csv(self._output_path + "n_gram_" + str(current_n) + "_count.csv", index=False)

                    del keys, values, n_gram_size_array, dict, df
            print()
            self._db.add_author_features_fast(author_features)

    def extract_and_clean_content(self, record):
        content = record.content
        content = clean_tweet(content)
        words = content.split()
        if self._remove_stopwords == True:
            words = [word for word in words if word not in stopwords.words('english')]
        if self._stemming == True:
            stemmer = PorterStemmer()
            words = stem_tokens(words, stemmer)
        return words

    def _update_dictionary(self, key, dict):
        if key not in dict:
            dict[key] = 1
        else:
            dict[key] += 1
