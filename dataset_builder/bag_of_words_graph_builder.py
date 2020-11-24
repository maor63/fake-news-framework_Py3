
import logging
from dataset_builder.graph_builder import GraphBuilder
import time
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.simple import SpaceTokenizer
from itertools import *

class GraphBuilder_Bag_Of_Words(GraphBuilder):
    """The edge between two authors is the jaccard similarity between the bag of words of each author"""

    def __init__(self, db):
        GraphBuilder.__init__(self, db)

        if self._domain == 'Microblog':
            self._tokenizer = TweetTokenizer()
        else:
            self._tokenizer = SpaceTokenizer()

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting posts from DB ")

        if self._num_of_random_authors_for_graph is None:
            posts_by_domain = self._db.get_author_posts_dict_by_minimal_num_of_posts(self._domain, self._min_number_of_posts_per_author)
        else:
            #if not self._are_already_randomize_authors_for_graphs():
            #    self._db.randomize_authors_for_graph(self._min_number_of_posts_per_author, self._domain, self._num_of_random_authors_for_graph)
            posts_by_domain = self._db.get_random_author_posts_dict_by_minimal_num_of_posts()

        all_authors_count = len(list(posts_by_domain.keys()))
        total_combinations = (all_authors_count * (all_authors_count - 1)) / 2
        current = 0
        # Dictionary: key = author_guid, value = list of posts
        bag_of_words_per_author = {}

        for author, posts in posts_by_domain.items():
            bow = []
            for post in posts:
				content = post.content
				if content is not None:
					bow += self._tokenizer.tokenize(content)
            bag_of_words_per_author[author] = frozenset(bow)
            current += 1
            if current % 10000 == 0:
                print('\r done author ' + str(current) + ' out of ' + str(all_authors_count), end='')
        logging.info("done computing bag of words ")
        all_pairs = combinations(list(bag_of_words_per_author.keys()),2)
        """
        Casting all_pairs to an iterable object (frozenset) is NOT a good idea since combinations function returns a generator object,
        which is more memory and CPU efficient than iterable objects
        """
        logging.info("computing similarity between bag of words ")
        author_connections = []

        current = 0
        for author_a, author_b in all_pairs:
            weight = self.compute_jaccard_index(bag_of_words_per_author[author_a], bag_of_words_per_author[author_b])
            author_connections.append((author_a,author_b,weight))
            current +=1
            if current % 10000 == 0:
                print('\r done pair '+str(current)+' out of '+str(total_combinations), end='')
                self._fill_author_connections(author_connections)
                author_connections = []
        self._fill_author_connections(author_connections)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(" total time taken "+str(duration))

    def compute_jaccard_index(self, set_1, set_2):
        n = len(set_1.intersection(set_2))
        return n / float(len(set_1) + len(set_2) - n)