
from dataset_builder.graph_builder import GraphBuilder
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.simple import SpaceTokenizer
import logging
from itertools import *
import re

__author__ = "Aviad Elyashar"

class Bag_Of_Words_Graph_Builder(GraphBuilder):
    """The edge between two authors is the jaccard similarity between the bag of words of each author"""

    def __init__(self, db):
        GraphBuilder.__init__(self, db)

        self._author_guid_posts_dict = {}
        self._author_guid_bag_of_words_dict = {}
        self._word_dict = {}

        if self._domain == 'Microblog':
            self._tokenizer = TweetTokenizer()
        else:
            self._tokenizer = SpaceTokenizer()

    def execute(self, window_start=None):
        pass

    def fill_author_guid_posts_dictionary(self):
        self._author_guid_posts_dict = self._db.get_author_posts_dict_by_minimal_num_of_posts(self._domain,
                                                                                 self._min_number_of_posts_per_author)

    def fill_author_guid_bag_of_words_dictionary(self):
        all_authors_count = len(list(self._author_guid_posts_dict.keys()))
        i = 0
        # Dictionary: key = author_guid, value = list of posts
        for author_guid, posts in self._author_guid_posts_dict.items():
            bow = []
            for post in posts:
                content = post.content
                content = content.lower()
                content = re.sub(r'http\S+', '', content)
                if content is not None:
                    bow += self._tokenizer.tokenize(content)

            bow = list(frozenset(bow))
            self._author_guid_bag_of_words_dict[author_guid] = bow

            for word in bow:
                if word not in self._word_dict:
                    self._word_dict[word] = word
            i += 1
            if i % 100000 == 0:
                print('\r done author ' + str(i) + ' out of ' + str(all_authors_count), end='')
        logging.info("done computing bag of words ")

    def compute_jaccard_index(self, set_1, set_2):
        n = len(set_1.intersection(set_2))
        return n / float(len(set_1) + len(set_2) - n)

    def fill_author_guid_bag_of_words_dictionary_and_calculate_all_combinations(self):
        self.fill_author_guid_bag_of_words_dictionary()

        author_guids = list(self._author_guid_bag_of_words_dict.keys())
        all_authors_count = len(author_guids)
        all_pairs = combinations(author_guids, 2)
        total_combinations = (all_authors_count * (all_authors_count - 1)) / 2
        """
        Casting all_pairs to an iterable object (frozenset) is NOT a good idea since combinations function returns a generator object,
        which is more memory and CPU efficient than iterable objects
        """
        logging.info("computing similarity between bag of words ")

        i = 0
        for author_guid_1, author_guid_2 in all_pairs:
            i += 1
            print('\r calculating pairs of authors : {0}/{1}'.format(i, total_combinations), end='')
            author_guid_1_bag_of_words = self._author_guid_bag_of_words_dict[author_guid_1]
            author_guid_2_bag_of_words = self._author_guid_bag_of_words_dict[author_guid_2]

            self.calculate_jaccard_index_create_and_save_connection(author_guid_1, author_guid_2, author_guid_1_bag_of_words,
                                                                    author_guid_2_bag_of_words)

        self._db.save_author_connections(self._author_connections_edges)

    def calculate_jaccard_index_create_and_save_connection(self, author_guid_1, author_guid_2,
                                                           author_guid_1_bag_of_words, author_guid_2_bag_of_words):

        weight = self.compute_jaccard_index(set(author_guid_1_bag_of_words), set(author_guid_2_bag_of_words))
        self._create_and_optional_save_connection(author_guid_1, author_guid_2, weight)

