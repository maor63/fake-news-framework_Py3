# Created by jorgeaug at 20/06/2016
# placeholder for sentiment analysis class
import re
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

class SentimentAnalysis:
    '''
    This module is responsible for prediction of sentiment labels for each post
    A sentiment label can have one of three posible values: positive, negative or neutral
    '''
    def __init__(self, db):
        self._db = db
        #self._posts = self._db.get_posts_by_domain(unicode('Microblog'))
        emoticons_str = r"""
         (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
        )"""

        regex_str = [emoticons_str, r'<[^>]+>', # HTML tags
                                                r'(?:@[\w_]+)', # @-mentions
                                                r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
                                                r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
                                                r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
                                                r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
                                                r'(?:[\w_]+)', # other words
                                                r'(?:\S)' # anything else
                                            ]
        self._tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
        self._emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

    def setUp(self):
        pass

    def execute(self, window_start=None):
        self.train_model()
        #for post in self._posts:
        #    tweet = post.content
        #    tokens = self.preprocess(tweet)
        #    # Create a list with all the terms
        #    terms_all = [term for term in tokens]



    def tokenize(self, text):
        return self.tokens_re.findall(text)

    def preprocess(self, text, lowercase=False):
        tokens = self.tokenize(text)
        if lowercase:
            tokens = [token if self._emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens

    def train_model(self):
        n_instances = 100
        subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
        obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

        train_subj_docs = subj_docs[:80]
        test_subj_docs = subj_docs[80:100]
        train_obj_docs = obj_docs[:80]
        test_obj_docs = obj_docs[80:100]
        training_docs = train_subj_docs+train_obj_docs
        testing_docs = test_subj_docs+test_obj_docs

        sentim_analyzer = SentimentAnalyzer()
        all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

        unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
        sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

        training_set = sentim_analyzer.apply_features(training_docs)
        test_set = sentim_analyzer.apply_features(testing_docs)

        trainer = NaiveBayesClassifier.train
        classifier = sentim_analyzer.train(trainer, training_set)
        #Training classifier
        for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
            print(('{0}: {1}'.format(key, value)))


