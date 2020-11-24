# def evaluate_trec2012(self):
#     topics = read_topics('2012.topics.MB51-110.txt')
#     topic_judgments = read_judgments('adhoc-qrels')
#     topic_candidates = extract_candidates_from_judgments(topic_judgments)
from sklearn.feature_extraction.text import TfidfVectorizer
from sortedcollections import OrderedSet

from DB.schema_definition import *
from commons.method_executor import Method_Executor
from preprocessing_tools.abstract_controller import AbstractController
from collections import namedtuple, defaultdict
import xmltodict
import csv
import parse
from dateutil import parser
from commons.commons import *
import pandas as pd

from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api


class Trec2012MicroblogTrackImporter(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._topics_path = self._config_parser.eval(self.__class__.__name__, "topics_path")
        self._judgment_path = self._config_parser.eval(self.__class__.__name__, "judgment_path")
        self._num_of_relevant_tweets = self._config_parser.eval(self.__class__.__name__, "num_of_relevant_tweets")
        self._num_of_description_words = self._config_parser.eval(self.__class__.__name__, "num_of_description_words")
        self._twitter_api = Twitter_Rest_Api(db)

    def load_data(self):
        topics = self._read_trec_topics(self._topics_path)
        topic_judgments = self._read_judgments(self._judgment_path)
        claims = self._extract_claims_from_judgments(topics, topic_judgments)
        self._db.addPosts(claims)
        self._create_tweet_corpus_from_judgments(self._judgment_path)

    def set_description_from_relevant(self):
        claims = self._db.get_claims()
        topic_judgments = self._read_judgments(self._judgment_path)
        posts = self._db.get_posts()
        post_dict = {p.post_id: p for p in posts}
        for claim in claims:
            topic_id = int(claim.claim_id)
            # tweets = self._twitter_api.get_tweets_by_ids(topic_judgments[topic_id][:self._num_of_relevant_tweets])
            # posts, authors = self._db.convert_tweets_to_posts_and_authors(tweets, self._domain)
            posts = list(map(post_dict.get, topic_judgments[topic_id][:self._num_of_relevant_tweets]))
            claim_content = OrderedSet(claim.keywords.lower().split())
            for post in posts:
                list(map(claim_content.add, clean_tweet(post.content).split()))
                # if len(claim_content) > 25:
                #     break
            claim.description = clean_claim_description(' '.join(claim_content), True)
        self._db.addPosts(claims)

    def set_description_from_tf_idf_best(self):
        tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
        posts = self._db.get_posts()
        corpus = [clean_claim_description(p.content, True) for p in posts]
        tf_idf_vectorizer.fit_transform(corpus)
        word_tf_idf_dict = defaultdict(float, list(zip(tf_idf_vectorizer.get_feature_names(), tf_idf_vectorizer.idf_)))
        post_dict = {p.post_id: p for p in posts}
        topic_judgments = self._read_judgments(self._judgment_path)

        claims = self._db.get_claims()
        for i, claim in enumerate(claims):
            init_query_words = set(claim.keywords.lower().split())
            claim_content = set()
            relevant_posts_ids = topic_judgments[int(claim.claim_id)]
            words = set()
            for post_id in relevant_posts_ids:
                if post_id in post_dict:
                    words.update(clean_claim_description(post_dict.get(post_id).content, True).split())
            best_words = sorted(words, key=lambda k: word_tf_idf_dict[k], reverse=True)[
                         :self._num_of_description_words + len(init_query_words)]
            claim_content.update(best_words)
            pass
            claim_content = claim_content - init_query_words
            claim_description = clean_claim_description(' '.join(claim_content), True)
            claim.description = ' '.join(claim_description.split())

        self._db.addPosts(claims)

    def _read_trec_topics(self, topics_path):
        trec_topic_fields = ['num', 'query', 'querytime', 'querytweettime']
        TrecTopic = namedtuple('TrecTopic', trec_topic_fields)
        topic_file = open(topics_path)
        trec_topics = []
        for topic_xml in topic_file.read().split('\n\n')[:-1]:
            trec_topic_dict = xmltodict.parse(topic_xml)
            trec_topic = TrecTopic._make([trec_topic_dict['top'][field] for field in trec_topic_fields])
            trec_topics.append(trec_topic)
        return trec_topics

    # def _read_judgments(self, judgment_path):
    #     judgments = pd.read_csv(judgment_path, sep=' ', names=['topic_id', 'Q', 'tweet_id', 'rel'])
    #     tweets = self._twitter_api.get_tweets_by_ids(judgments['tweet_id'], pre_save=False)
    #     topic_tweet_id_dict = dict(judgments[['topic_id', 'tweet_id']].to_records(index=False))
    #     tweet_id_rel_dict = dict(judgments[['tweet_id', 'rel']].to_records(index=False))
    #     posts, authors = self._db.convert_tweets_to_posts_and_authors(tweets, u'Trec2012')

    def _read_judgments(self, judgment_path):
        topic_high_relevant_judgment_dict = defaultdict(list)
        topic_relevant_judgment_dict = defaultdict(list)

        for topic, Q, docid, rel in csv.reader(open(judgment_path, "rb"), delimiter=' '):
            if int(rel) > 1:
                topic_high_relevant_judgment_dict[int(topic)].append(docid)
            elif int(rel) == 1:
                topic_relevant_judgment_dict[int(topic)].append(docid)
        for topic_id, tweet_ids in topic_relevant_judgment_dict.items():
            # if topic_id not in topic_high_relevant_judgment_dict:
            topic_high_relevant_judgment_dict[topic_id].extend(tweet_ids)
        return topic_high_relevant_judgment_dict

    def _extract_claims_from_judgments(self, topics, topic_judgments):
        claims = []
        for trec_topic in topics:
            # tweet_id = tweet_ids[0]
            topic_id = int(parse.parse('Number: MB{}', trec_topic.num)[0])
            # tweets = self._twitter_api.get_tweets_by_ids(topic_judgments[topic_id][:10])
            # posts, authors = self._db.convert_tweets_to_posts_and_authors(tweets, self._domain)
            claim_content = set(trec_topic.query.split())
            # for post in []:
            #     claim_content.update(clean_tweet(post.content).split())
            #     if len(claim_content) > 25:
            #         break
            claim = self._convet_trec_topic_to_claim(' '.join(claim_content), topic_id, trec_topic)
            claims.append(claim)
        return claims

    def _convet_trec_topic_to_claim(self, claim_content, topic_id, trec_topic):
        claim = Claim()
        claim.claim_id = topic_id
        claim.verdict_date = parser.parse(trec_topic.querytime).date()
        claim.domain = 'Trec2012'
        claim.title = trec_topic.query
        claim.keywords = trec_topic.query
        claim.description = claim_content
        return claim

    def _create_tweet_corpus_from_judgments(self, judgment_path):
        judgment_df = pd.read_csv(judgment_path, delimiter=' ', names=['topic', 'Q', 'docid', 'rel'], )
        tweet_ids = judgment_df['docid'].tolist()
        tweets = self._twitter_api.get_tweets_by_ids(tweet_ids, pre_save=False)
        posts, authors = self._db.convert_tweets_to_posts_and_authors(tweets, 'Trec2012')
        claim_tweet_connections = []
        for post in posts:
            post.post_id = str(post.post_osn_id)
        post_osn_id_posts_dict = set(p.post_osn_id for p in posts)
        for topic_id, post_osn_id in judgment_df[['topic', 'docid']].to_records(index=False):
            if post_osn_id in post_osn_id_posts_dict:
                claim_tweet_connection = Claim_Tweet_Connection()
                claim_tweet_connection.claim_id = str(topic_id)
                claim_tweet_connection.post_id = str(post_osn_id)
                claim_tweet_connections.append(claim_tweet_connection)
        self._db.addPosts(claim_tweet_connections)
        self._db.addPosts(posts)
        judgment_df[judgment_df['docid'].isin(post_osn_id_posts_dict)].to_csv(judgment_path + '_filtered', sep=' ',
                                                                              header=False, index=False)
