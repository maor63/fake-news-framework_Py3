import numpy as np
from DB.schema_definition import Claim_Keywords_Connections
from commons.method_executor import Method_Executor
from commons.commons import *
import random
import os
from collections import Counter, defaultdict
from nltk.tag import StanfordNERTagger, pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordsGenerator(Method_Executor):

    def __init__(self, db):
        super(KeywordsGenerator, self).__init__(db)
        self._random_keywords_size = self._config_parser.eval(self.__class__.__name__, "random_keywords_size")
        self._remove_stop_words = self._config_parser.eval(self.__class__.__name__, "remove_stop_words")
        self._number_of_random_generations = self._config_parser.eval(self.__class__.__name__,
                                                                      "number_of_random_generations")
        self._tf_idf_vectorizer = None
        self._word_to_rank_mapping = {}

    def generate_manual_keywords(self):
        self._base_claim_key_words_generator('manual', self._generate_manual_keywords)

    def generate_random_keywords(self):
        for i in range(1, self._number_of_random_generations + 1):
            type_name = 'random_{0}_keywords_{1}'.format(self._random_keywords_size, i)
            self._base_claim_key_words_generator(type_name, self._generate_random_sample)

    def generate_equal_to_manual_random_keywords(self):
        for i in range(1, self._number_of_random_generations + 1):
            if self._remove_stop_words:
                type_name = 'random_equal_to_manual_{0}_{1}'.format('remove_stopwords', i)
            else:
                type_name = 'random_equal_to_manual_{0}_{1}'.format('with_stopwords', i)
            self._base_claim_key_words_generator(type_name, self._generate_random_sample_with_same_size_as_manual)

    def generate_post_tagging_keywords(self):
        keywords_name = 'pos_tagging_size_{}'.format(self._random_keywords_size)
        self._base_claim_key_words_generator(keywords_name, self._generate_post_tagging_keywords)

    def generate_tf_idf_keywords(self):
        keywords_name = 'tf_idf_size_{}'.format(self._random_keywords_size)
        self._base_claim_key_words_generator(keywords_name, self._generate_tf_idf_keywords)

    def get_word_tf_idf_of_claims(self):
        if self._remove_stop_words:
            self._tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self._tf_idf_vectorizer = TfidfVectorizer()
        claims = self._db.get_claims()
        corpus = [claim.description for claim in claims]
        tf = self._tf_idf_vectorizer.fit_transform(corpus)
        word_tf_idf_dict = defaultdict(float, list(zip(self._tf_idf_vectorizer.get_feature_names(), self._tf_idf_vectorizer.idf_)))
        return word_tf_idf_dict

    def _base_claim_key_words_generator(self, type_name, generate_keywords_fn):
        claims = self._db.get_claims()
        connections = []
        for claim in claims:
            claim_keywords_connections = Claim_Keywords_Connections()
            claim_id = claim.claim_id
            keywords = generate_keywords_fn(claim)
            claim_keywords_connections.claim_id = claim_id
            claim_keywords_connections.keywords = keywords
            claim_keywords_connections.type = type_name
            connections.append(claim_keywords_connections)
        self._db.addPosts(connections)

    def _generate_manual_keywords(self, claim):
        return claim.keywords

    def _generate_random_sample(self, claim):
        description = clean_claim_description(claim.description, self._remove_stop_words)
        description_split = description.split(' ')
        return ' '.join(random.sample(description_split, min(self._random_keywords_size, len(description_split))))

    def _generate_random_sample_with_same_size_as_manual(self, claim):
        description = clean_claim_description(claim.description, self._remove_stop_words)
        description_split = description.split(' ')
        manual_keywords_size = len((claim.keywords.split('||')[0]).split(' '))
        return ' '.join(
            random.sample(description_split, min(manual_keywords_size, len(description_split))))

    def _generate_post_tagging_keywords(self, claim):
        word_rank_dict = self.get_word_pos_tagging_dict(claim)
        keywords_size = self._random_keywords_size
        return self._generate_keywords_by_rank_and_size(claim, keywords_size, word_rank_dict)

        # description = claim.description
        # claim_words = word_tokenize(description)
        # keywords = []
        # word_tag_tuples = pos_tag(claim_words)
        # name = []
        # for i, word_tag_tuple in enumerate(word_tag_tuples):
        #     word, tag = word_tag_tuple
        #     if tag == 'NNP':
        #         name.append(word)
        #     else:
        #         if len(name) > 0:
        #             keywords.append(' '.join(name))
        #             name = []
        #         if 'NN' in tag:
        #             keywords.append(word)
        #         elif i + 2 < len(word_tag_tuples) and 'JJ' in tag and 'NN' in word_tag_tuples[i + 1][1]:
        #             keywords.append(word)
        #         elif i + 3 < len(word_tag_tuples) and 'RB' in tag and 'JJ' in word_tag_tuples[i + 1][1] and 'NN' in \
        #                 word_tag_tuples[i + 2][1]:
        #             keywords.append(word)
        #         elif i > 0 and 'CD' in tag and 'NN' in word_tag_tuples[i - 1][1]:
        #             keywords.append(word)
        #
        # return ' '.join(keywords)

    def get_word_pos_tagging_dict(self, claim):
        word_pos_tagging_rank_dict = defaultdict(float)
        pos_to_rank = defaultdict(float)
        pos_to_rank['NOUN'] = 1.0
        pos_to_rank['ADJ'] = 0.75
        pos_to_rank['ADV'] = 0.5
        pos_to_rank['NUM'] = 0.25
        description = claim.description
        claim_words = word_tokenize(description)

        word_tag_tuples = pos_tag(claim_words, tagset='universal')
        for word, tag in word_tag_tuples:
            word_pos_tagging_rank_dict[word.lower()] = pos_to_rank[tag]

        return word_pos_tagging_rank_dict

    def _generate_tf_idf_keywords(self, claim):
        word_rank_dict = self.get_word_tf_idf_of_claims()
        keywords_size = self._random_keywords_size
        return self._generate_keywords_by_rank_and_size(claim, keywords_size, word_rank_dict)

    def _generate_keywords_by_rank_and_size(self, claim, keywords_size, word_rank_dict):
        description = clean_claim_description(claim.description, self._remove_stop_words)
        description_split = description.split(' ')
        if keywords_size == 0 or keywords_size == 'manual':
            keywords_size = len((claim.keywords.split(',')[0]).split(' '))
        else:
            keywords_size = min(keywords_size, len(description_split))
        words_to_tf_idf = Counter(
            {word.lower(): word_rank_dict[word.lower()] for word in description_split if
             word.lower() in word_rank_dict})
        most_commons = words_to_tf_idf.most_common(min(keywords_size, len(words_to_tf_idf)))
        return ' '.join([word for word, rank in most_commons])
