# -*- coding: utf-8 -*-

import csv
import math
import random
from collections import defaultdict

import pandas as pd
from gensim.models import FastText
# from gensim.models.wrappers import FastText
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from DB.schema_definition import Claim, Post, Claim_Tweet_Connection
from commons.commons import *
from dataset_builder.feature_extractor.feature_argument_parser import ArgumentParser
# from dataset_builder.feature_extractor.lda_topic_feature_generator import LDATopicFeatureGenerator
from dataset_builder.information_retrieval_models.BM25_model import BM25Model
from dataset_builder.information_retrieval_models.LSI import LSI_m, build_term_tf_idf_model, build_term_count_model
from dataset_builder.information_retrieval_models.TF_IDF_ranker import TF_IDF_Ranker
from dataset_builder.word_embedding.glove_word_embedding_model_creator import GloveWordEmbeddingModelCreator



class KeywordEvaluator(ArgumentParser):
    def __init__(self, db):
        super(KeywordEvaluator, self).__init__(db)
        self._posts_for_labeling = self._config_parser.eval(self.__class__.__name__, "posts_for_labeling")
        self._interactive_iterations = self._config_parser.eval(self.__class__.__name__, "interactive_iterations")
        self._random_claim_size = self._config_parser.eval(self.__class__.__name__, "random_claim_size")
        self._output_keywords_counts = self._config_parser.eval(self.__class__.__name__, "output_keywords_count")
        self._fasttext_trained_model_path = self._config_parser.eval(self.__class__.__name__,
                                                                     "fasttext_trained_model_path")
        self._use_word_embedding_table = self._config_parser.eval(self.__class__.__name__, "use_word_embedding_table")
        self._word_embedding_table = self._config_parser.eval(self.__class__.__name__, "word_embedding_table")
        self._word_vector_dict = {}
        self._distance_metric = self._config_parser.eval(self.__class__.__name__, "distance_metric")
        self._keywords_type_for_evaluation = self._config_parser.eval(self.__class__.__name__,
                                                                      "keywords_type_for_evaluation")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._words_not_in_glove = 0
        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")
        self._relevance_score_method = self._config_parser.eval(self.__class__.__name__, "relevance_score_method")
        self._min_tweet_count = self._config_parser.eval(self.__class__.__name__, "min_tweet_count")
        self._word_post_dictionary = defaultdict(set)
        self._post_dictionary = {}
        self._model = None

    def setUp(self):
        pass


        if self._use_word_embedding_table:
            if not self._db.is_table_exist(self._word_embedding_table):
                glove_loader = GloveWordEmbeddingModelCreator(self._db)
                glove_loader._load_wikipedia_300d_glove_model()
            self._word_vector_dict = self._db.get_word_vector_dictionary(self._word_embedding_table)
        else:
            model = FastText.load_fasttext_format(self._fasttext_trained_model_path)
            self._model = model
            self._word_vector_dict = model.wv
        path = os.path.join(self._output_path, 'claims_{}'.format(self._distance_metric))
        if not os.path.exists(path):
            os.makedirs(path)

    def execute(self, window_start=None):
        for action_name in self._actions:
            try:
                getattr(self, action_name)()
            except AttributeError as e:
                print('\nError: {0}\n'.format(e.message), file=sys.stderr)
                logging.error('Error: {0}'.format(e.message))

    def evaluate_using_word_embedding(self):
        # self._set_up_word_mbedding_model()
        claims = self._db.get_claims()
        keywords_to_result = {}
        claim_ids_set_for_keywords = defaultdict(set)
        for keywords_type in self._keywords_type_for_evaluation:
            claim_id_distance_and_tweet_num = self.evaluate_keywords(keywords_type)
            keywords_to_result[keywords_type] = claim_id_distance_and_tweet_num
            keyword_type_records = []
            claims = [claim for claim in claims if claim.claim_id in claim_id_distance_and_tweet_num]
            for i, claim in enumerate(claims):
                print("\r eval keywords to claim {0}/{1}".format(i, len(claims)), end='')
                claim_id = claim.claim_id
                description = claim.description
                data = claim_id_distance_and_tweet_num[claim_id]
                keyword_type_records.append([claim_id, description, keywords_type, data['distance_with_keywords'],
                                             data['distance_without_keywords'], data['tweet_num']])
                if data['tweet_num'] > 0:
                    claim_ids_set_for_keywords[keywords_type].add(claim_id)
            keywords_df = pd.DataFrame(keyword_type_records,
                                       columns=['claim_id', 'description', 'keywords_type', 'distance_with_keywords',
                                                'distance_without_keywords', 'tweet_num'])
            keywords_df.to_csv(os.path.join(self._output_path, 'keywords_{}_evaluation_{}.csv'.format(keywords_type,
                                                                                                      self._distance_metric)))
        print()
        self._print_keywords_compare_csv(claim_ids_set_for_keywords, keywords_to_result)
        self._print_mtutual_compare_csv(claim_ids_set_for_keywords, keywords_to_result)

    def _print_mtutual_compare_csv(self, claim_ids_set_for_keywords, keywords_to_result):
        f = open(os.path.join(self._output_path, 'keywords_mutual_compare_{}.csv'.format(self._distance_metric)), 'wb')
        f.write(
            'keywords_type1, keywords_type2, common_claims_count, distance_with_keywords1, distance_with_keywords2, ' +
            'avg_tweets1, avg_tweets2\n')
        keywords_list = list(keywords_to_result.keys())
        for i, keywords_type1 in enumerate(keywords_list):
            for j, keywords_type2 in enumerate(keywords_list):
                if i < j:
                    claim_ids_set1 = claim_ids_set_for_keywords[keywords_type1]
                    claim_ids_set2 = claim_ids_set_for_keywords[keywords_type2]
                    combined_claim_ids = claim_ids_set1 & claim_ids_set2
                    claim_id_distance_and_tweet_num1 = keywords_to_result[keywords_type1]
                    claim_id_distance_and_tweet_num2 = keywords_to_result[keywords_type2]
                    avg_distance1 = numpy.mean(
                        [data['distance_with_keywords'] for claim_id, data in
                         claim_id_distance_and_tweet_num1.items() if
                         claim_id in combined_claim_ids])
                    avg_distance2 = numpy.mean(
                        [data['distance_with_keywords'] for claim_id, data in
                         claim_id_distance_and_tweet_num2.items() if
                         claim_id in combined_claim_ids])
                    avg_tweets1 = numpy.mean(
                        [data['tweet_num'] for claim_id, data in claim_id_distance_and_tweet_num1.items() if
                         claim_id in combined_claim_ids])
                    avg_tweets2 = numpy.mean(
                        [data['tweet_num'] for claim_id, data in claim_id_distance_and_tweet_num2.items() if
                         claim_id in combined_claim_ids])

                    f.write(
                        '{}, {}, {}, {}, {}, {}, {}\n'.format(keywords_type1, keywords_type2, len(combined_claim_ids),
                                                              avg_distance1, avg_distance2, avg_tweets1, avg_tweets2))
        f.flush()
        f.close()

    def _print_keywords_compare_csv(self, claim_ids_set_for_keywords, keywords_to_result):
        f = open(os.path.join(self._output_path, 'keywords_compare_{}.csv'.format(self._distance_metric)), 'wb')
        # f.write('common_claims, {}\n'.format(len(claim_ids_set_for_compare)))
        f.write('keywords_type, distance_with_keywords, distance_without_keywords, avg_tweets, '
                'claims_with_posts_by_keywords, total_num_of_tweets\n')
        claims_count = float(len(self._db.get_claims()))
        for keywords_type in keywords_to_result:
            total_tweets = len(self._db.get_posts_by_selected_domain(keywords_type))
            claim_ids_set = claim_ids_set_for_keywords[keywords_type]
            claim_id_distance_and_tweet_num = keywords_to_result[keywords_type]
            avg_distance_with_keywords = numpy.mean(
                [data['distance_with_keywords'] for claim_id, data in claim_id_distance_and_tweet_num.items() if
                 claim_id in claim_ids_set])
            avg_distance_without_keywords = numpy.mean(
                [data['distance_without_keywords'] for claim_id, data in claim_id_distance_and_tweet_num.items() if
                 claim_id in claim_ids_set])
            avg_tweets = numpy.sum(
                [data['tweet_num'] for claim_id, data in claim_id_distance_and_tweet_num.items() if
                 claim_id in claim_ids_set])
            avg_tweets = avg_tweets / claims_count
            f.write('{}, {}, {}, {}, {}, {}\n'.format(keywords_type, avg_distance_with_keywords,
                                                      avg_distance_without_keywords, avg_tweets, len(claim_ids_set),
                                                      total_tweets))
        f.flush()
        f.close()

    def evaluate_keywords(self, keywords_type):
        claims = self._db.get_claims()
        if keywords_type == 'total_random':
            posts = self._db.get_posts()
            source_id_elements = {claim.claim_id: random.sample(posts, min(300, len(posts))) for claim in claims}
            claim_id_keywords_dict = {claim.claim_id: '' for claim in claims}
            pass
        else:
            args = self.generate_args_for_keywords(keywords_type)
            source_id_elements = self._get_source_id_target_elements(args)
            claim_id_keywords_dict = self._db.get_claim_id_keywords_dict_by_connection_type(keywords_type)
        claim_id_distance_and_tweet_num = defaultdict(dict)
        # claims = [claim for claim in claims if claim.claim_id in claim_id_keywords_dict]
        for i, claim in enumerate(claims):
            print('\rprocess claim {}/{}'.format(str(i + 1), len(claims)), end='')
            claim_description_words = clean_claim_description(claim.description, False).split(' ')
            posts = source_id_elements[claim.claim_id]
            distances_with_keywords = []
            distances_without_keywords = []
            keywords = claim_id_keywords_dict.get(claim.claim_id, '')
            calim_post_anliysis_rows = []
            for post in posts:
                post_words = clean_claim_description(post.content, False).split(' ')
                distance_with_keywords, distance_without_keywords = self._min_distance(claim_description_words,
                                                                                       post_words, keywords)
                distances_with_keywords.append(distance_with_keywords)
                distances_without_keywords.append(distance_without_keywords)

                calim_post_anliysis_rows.append(
                    [claim.claim_id, claim.description, post.post_id, clean_claim_description(post.content, False),
                     keywords_type,
                     distance_with_keywords,
                     distance_without_keywords, post.post_type])
            pd.DataFrame(calim_post_anliysis_rows,
                         columns=['claim_id', 'description', 'post_id', 'content', 'keywords_type',
                                  'distance_with_keywords',
                                  'distance_without_keywords', 'class']).to_excel(
                os.path.join(self._output_path, 'claims_{}'.format(self._distance_metric),
                             'claim_{}_{}.xlsx'.format(keywords_type, i)))
            if len(distances_with_keywords) > 0:
                distances_with_keywords = [dist for dist in distances_with_keywords if dist != -1]
                distances_without_keywords = [dist for dist in distances_without_keywords if dist != -1]
                claim_id_distance_and_tweet_num[claim.claim_id] = {
                    'distance_with_keywords': float(numpy.mean(distances_with_keywords)),
                    'distance_without_keywords': float(numpy.mean(distances_without_keywords)),
                    'tweet_num': len(distances_with_keywords)}
            else:
                claim_id_distance_and_tweet_num[claim.claim_id] = {'distance_with_keywords': 1000,
                                                                   'distance_without_keywords': 1000,
                                                                   'tweet_num': len(distances_with_keywords)}
        print()
        return claim_id_distance_and_tweet_num

    def evaluate_keywords_for_claim(self, keywords_type, claim):
        source_id_elements = self._get_claim_posts_dict(claim, keywords_type)
        # self._db.get_claim_id_keywords_dict_by_connection_type(keywords_type)
        posts = source_id_elements[claim.claim_id]
        claim_id_keywords_dict = self._db.get_claim_id_keywords_dict_by_connection_type(keywords_type)
        keywords = claim_id_keywords_dict[claim.claim_id]
        return self.eval_claim_tweets(claim.description, keywords, posts)

    def _get_claim_posts_dict(self, claim, keywords_type):
        args = {
            'source': {'table_name': 'claims', 'id': 'claim_id',
                       "where_clauses": [{"field_name": 'claim_id', "value": str(claim.claim_id)}]},
            'connection': {'table_name': 'claim_tweet_connection',
                           'source_id': 'claim_id',
                           'target_id': 'post_id',
                           "where_clauses": []},
            'destination': {'table_name': 'posts', 'id': 'post_id',
                            'target_field': 'content',
                            "where_clauses": [{"field_name": 'domain', "value": keywords_type}]}
        }
        source_id_elements = self._get_source_id_target_elements(args)
        return source_id_elements

    def eval_claim_tweets(self, claim_description, keywords, posts):
        claim_description_words = clean_claim_description(claim_description, True).split(' ')
        distances = []
        for post in posts:
            post_words = clean_claim_description(post.content, True).split(' ')
            distance_with_keywords, distance_without_keywords = self._min_distance(claim_description_words, post_words,
                                                                                   keywords)
            distances.append(distance_with_keywords)
        if len(distances) > 0:
            return {'distance': float(numpy.mean(distances)), 'tweet_num': len(distances)}
        else:
            return {'distance': 1000, 'tweet_num': len(distances)}

    def generate_args_for_keywords(self, keywords_type):
        args = {
            'source': {'table_name': 'claims', 'id': 'claim_id',
                       "where_clauses": []},
            'connection': {'table_name': 'claim_tweet_connection',
                           'source_id': 'claim_id',
                           'target_id': 'post_id',
                           "where_clauses": []},
            'destination': {'table_name': 'posts', 'id': 'post_id',
                            'target_field': 'content',
                            "where_clauses": [{"field_name": 'domain', "value": keywords_type}]}
        }
        return args

    def _min_distance(self, claim_description_words, post_words, keywords):
        distance_method = getattr(self, self._relevance_score_method)
        avg_with_keywords = distance_method(claim_description_words, post_words)
        if keywords != '':
            keywords = set(keywords.split(' '))
            claim_description_words = set(claim_description_words) - keywords
        # if len(claim_description_words) == 0 or len(post_words) == 0:
        #     print('claim words: {}, post words {}'.format(len(claim_description_words), len(post_words)))
        avg_without_keywords = distance_method(claim_description_words, post_words)
        return avg_with_keywords, avg_without_keywords

    def get_DESM_min_distance(self, claim_description_words, post_words):
        claim_words_vectors = self._get_words_vectors(claim_description_words)
        claim_words_centroid = claim_words_vectors.mean(axis=0).reshape(1,-1)
        post_words_vectors = self._get_words_vectors(post_words)
        try:
            if self._distance_metric != 'dot':
                return cdist(post_words_vectors, claim_words_centroid, metric=self._distance_metric).mean()
            else:
                return np.dot(post_words_vectors, np.transpose(claim_words_centroid)).mean()
        except Exception as e:
            print(e, len(claim_description_words), len(post_words))
            return -1

    def get_avg_min_distance(self, claim_description_words, post_words):
        claim_post_words_distances = self._get_words_min_distances(claim_description_words, post_words)
        return np.mean(claim_post_words_distances)

    def get_avg_squared_min_distance(self, claim_description_words, post_words):
        claim_post_words_distances = self._get_words_min_distances(claim_description_words, post_words)
        return np.mean(claim_post_words_distances ** 2)

    def _get_words_min_distances(self, claim_description_words, post_words):
        claim_words_vectors = self._get_words_vectors(claim_description_words)
        post_words_vectors = self._get_words_vectors(post_words)
        try:
            # ax = 1 if len(claim_description_words) >= len(post_words) else 0
            if self._distance_metric != 'dot':
                return cdist(post_words_vectors, claim_words_vectors, metric=self._distance_metric).min(axis=1)
            else:
                return np.dot(post_words_vectors, np.transpose(claim_words_vectors)).min(axis=1)
        except Exception as e:
            return -1

    def _get_words_vectors(self, words):
        return np.array([self._word_vector_dict[word] for word in words if word in self._word_vector_dict])

    def build_word_post_dict_for_trec(self):
        self._word_post_dictionary = defaultdict(set)
        print("Load posts dictionary")
        self._post_dictionary = self._db.get_post_dictionary()
        posts = self._db.get_posts_filtered_by_domain('Trec2012')
        print("create words corpus")
        for i, post in enumerate(posts):
            print("\rprocess post {}/{}".format(str(i + 1), len(posts)), end='')
            for word in post.content.lower().split(' '):
                self._word_post_dictionary[word].add(post.post_id)
        print()

    def get_posts_from_word_post_dict(self, words, claim):
        post_sets = [self._word_post_dictionary[word] for word in words]
        if post_sets:
            result_post_ids = set.intersection(*post_sets)
            end_date = claim.verdict_date
            return [self._post_dictionary[post_id] for post_id in result_post_ids if
                    self._post_dictionary[post_id].date <= end_date]
        else:
            return []

    def trec_2012_output_evaluation(self):
        self.build_word_post_dict_for_trec()
        claim_post_score_dict = {}
        for keyword_count in self._output_keywords_counts:
            for keyword_type in self._keywords_type_for_evaluation:
                rows_min_count = []
                other_rows = []
                claims = self._db.get_claims_by_domain('Trec2012')
                claim_id_keywords_dict = self._db.get_claim_id_keywords_dict_by_connection_type(keyword_type)
                for i, claim in enumerate(claims):
                    # print('\r eval claim {}/{}'.format(str(i + 1), len(claims)), end='')
                    if claim.claim_id in claim_id_keywords_dict:
                        keywords_str = claim_id_keywords_dict[claim.claim_id]
                        posts = []
                        post_id_set = set()

                        for keywords in keywords_str.split('||')[:keyword_count]:
                            # result_posts = self._db.get_posts_from_domain_contain_words(u'Trec2012', keywords.split())
                            result_posts = self.get_posts_from_word_post_dict(keywords.split(), claim)
                            for post in result_posts:
                                if post.post_id not in post_id_set:
                                    post_id_set.add(post.post_id)
                                    posts.append(post)
                    else:
                        keywords_str = claim.keywords
                        # posts = self._db.get_posts_from_domain_contain_words(u'Trec2012', keywords_str.split())
                        posts = self.get_posts_from_word_post_dict(keywords_str.split(), claim)
                    for j, post in enumerate(posts):
                        msg = '\rcompute distance for claim {}/{}, post {}/{}, keyword_count {}'
                        print(msg.format(str(i + 1), len(claims), str(j + 1), len(posts), keyword_count), end='')
                        key = claim.claim_id + str(post.post_osn_id)
                        if key in claim_post_score_dict:
                            score = claim_post_score_dict[key]
                        else:
                            score, _ = self._min_distance(claim.description, post.content, keywords_str)
                            claim_post_score_dict[key] = score
                        if len(posts) >= self._min_tweet_count:
                            rows_min_count.append((claim.claim_id, post.post_osn_id, score, keyword_type))
                        else:
                            other_rows.append((claim.claim_id, post.post_osn_id, score, keyword_type))
                print()
                output_path = 'trec2012_output_{}_keywords_count_{}.txt'.format(keyword_type, keyword_count)

                rows = self._sort_rows(rows_min_count) + self._sort_rows(other_rows)
                pd.DataFrame(rows).to_csv(os.path.join(self._output_path, output_path), sep=' ', index=False,
                                          header=None)
        table = pd.read_sql_table('claim_keywords_connections', self._db.engine)
        table.to_excel(os.path.join(self._output_path, 'claim_keywords_connections_table.xlsx'))

    def _sort_rows(self, rows):
        rows = sorted(rows, key=lambda x: x[2])
        for row in list(rows):
            if row[2] == -1.0:
                rows.append(rows.pop(0))
            elif row[2] >= 0:
                break
        return rows

    def trec_2012_baseline(self):
        rows = []
        claims = self._db.get_claims_by_domain('Trec2012')

        for i, claim in enumerate(claims):
            posts = self.get_posts_from_word_post_dict(claim.keywords.lower().split(), claim)
            for j, post in enumerate(posts):
                print('\rcompute distance for claim {}/{}, post {}/{}'.format(str(i + 1), len(claims), str(j + 1),
                                                                              len(posts)), end='')
                # score, _ = self._min_distance(claim.description, post.content, clean_tweet(claim.keywords))
                score, _ = 0, 0
                rows.append((claim.claim_id, post.post_osn_id, score, 'baseline'))
        pd.DataFrame(rows).to_csv('trec2012_output_baseline.txt', sep=' ', index=False, header=None)

    def isEnglish(self, s):
        import string
        # assert isinstance(s, str)
        for c in string.punctuation + string.whitespace:
            s = s.replace(c, '')
        return s.isalnum()

    def trec_2102_eval_RME(self):
        topic_doc_rel = self._get_topic_doc_rel_dict('data/input/trec_data/adhoc-qrels_filtered')

        claims = self._db.get_claims_by_domain('Trec2012')
        args = self.generate_args_for_keywords('Trec2012')
        source_id_elements = self._get_source_id_target_elements(args)
        random_posts = random.sample(self._db.get_posts(), self._random_claim_size)
        random_claim = clean_claim_description(' '.join([p.content for p in random_posts]), True)
        i = 0
        rows = []
        rows_with_random = []
        for claim in claims:
            claim_RME_eval_rows = []
            posts = source_id_elements[claim.claim_id]
            msg = '\rcompute distance for claim {}/{}'.format(str(i + 1), len(source_id_elements))
            claim_description_words = claim.description.lower().split()
            post_score_dict = self._get_posts_score_dict_for_claim(claim_description_words, posts, msg)
            post_score_from_random_dict = self._get_posts_score_dict_for_claim(random_claim.split(), posts, msg)
            sorted_posts = self._sort_posts_using_post_dict(posts, post_score_dict)
            for post in sorted_posts:
                rows.append((claim.claim_id, post.post_osn_id, post_score_dict[post.post_osn_id], 'eval_RME'))
                score_with_random = post_score_from_random_dict[post.post_osn_id]
                rows_with_random.append((claim.claim_id, post.post_osn_id, score_with_random,
                                         'eval_RME_with_random_claim_{}'.format(self._random_claim_size)))
                key = str(claim.claim_id) + str(post.post_osn_id)

                claim_RME_eval_rows.append(
                    (claim.claim_id, claim.description, post.post_osn_id, post.content,
                     post_score_dict[post.post_osn_id], post_score_from_random_dict[post.post_osn_id],
                     topic_doc_rel[key]))

            i += 1
            output_path = '{}_{}.xlsx'.format(claim.claim_id, self._relevance_score_method)
            pd.DataFrame(claim_RME_eval_rows,
                         columns=['topic', 'claim_description', 'post_id', 'post_content', 'post_dist_from_claim',
                                  'score_from_random',
                                  'rel']).to_excel(
                os.path.join(self._output_path, output_path))

        output_path = 'trec2012_eval_RME_{}.txt'.format(self._relevance_score_method)
        pd.DataFrame(rows).to_csv(os.path.join(self._output_path, output_path), sep=' ', index=False, header=None)

        output_path = 'trec2012_eval_RME_{}_with_random_claim_{}.txt'.format(self._relevance_score_method,
                                                                             self._random_claim_size)
        pd.DataFrame(rows_with_random).to_csv(os.path.join(self._output_path, output_path), sep=' ', index=False,
                                              header=None)

    def _sort_posts_using_post_dict(self, posts, post_score_dict, reverse=False):
        return sorted(posts, key=lambda post: post_score_dict[post.post_osn_id], reverse=reverse)

    def _get_posts_score_dict_for_claim(self, claim_description_words, posts, msg, irrelevant_val=1000.0):
        post_score_dict = {}
        for j, post in enumerate(posts):
            print(msg + ' , post {}/{}'.format(str(j + 1), len(posts)), end='')
            if post.post_format == 'en':
                post_words = clean_claim_description(post.content, True).split()
                score, _ = self._min_distance(claim_description_words, post_words, '')
                if score == -1.0:
                    score = irrelevant_val
            else:
                score = irrelevant_val
            post_score_dict[post.post_osn_id] = score
        return post_score_dict

    def _get_topic_doc_rel_dict(self, qrels_file_path):
        judgment_df = pd.read_csv(qrels_file_path, delimiter=' ',
                                  names=['topic', 'Q', 'docid', 'rel'])
        topic_doc_rel = {}
        for topic_id, post_osn_id, rel in judgment_df[['topic', 'docid', 'rel']].to_records(index=False):
            key = str(topic_id) + str(post_osn_id)
            topic_doc_rel[key] = rel
        return topic_doc_rel

    def trec_2102_interactive_search_eval(self):
        # lda = LDATopicFeatureGenerator(self._db, **{'authors': [], 'posts': {}})
        self.build_word_post_dict_for_trec()
        topic_doc_rel = self._get_topic_doc_rel_dict('data/input/trec_data/adhoc-qrels_filtered')
        post_dictionary = self._db.get_post_dictionary()
        claims = self._db.get_claims_by_domain('Trec2012')
        args = self.generate_args_for_keywords('Trec2012')
        source_id_elements = self._get_source_id_target_elements(args)
        iteration_rows = [[]] * (self._interactive_iterations + 1)
        t_rows = []

        for i, claim in enumerate(claims):
            # if claim.claim_id not in ['110', '102', '92', '80', '67', '65', '69', '63', '61', '51']:
            #     continue
            claim_words = set(claim.keywords.lower().split())

            posts = source_id_elements[claim.claim_id]
            post_contents = [p.content for p in posts]
            if self._relevance_score_method.startswith('lsi'):
                if self._relevance_score_method == 'lsi_tf_idf':
                    print('LSI TF-IDF model')
                    doc_rank_model = LSI_m(post_contents, build_term_tf_idf_model)
                else:
                    print('LSI word count model')
                    doc_rank_model = LSI_m(post_contents, build_term_count_model)
            elif self._relevance_score_method == 'BM25':
                doc_rank_model = BM25Model(post_contents)
            elif self._relevance_score_method == 'tf_idf_ranker':
                doc_rank_model = TF_IDF_Ranker(post_contents)
            else:
                doc_rank_model = None
            irrelevant_words = set()
            labeled_posts = set()
            rel_posts = []
            irrel_posts = []
            add_rel = True
            for iteration in range(self._interactive_iterations + 1):

                msg = '\rcompute distance for claim {}/{}'.format(str(i + 1), len(posts))
                if doc_rank_model is not None:
                    post_rank = doc_rank_model.rank(' '.join(claim_words))
                    compund_score_dict = {p.post_osn_id:rank for p, rank in zip(posts, post_rank)}
                    sorted_topics = self._sort_posts_using_post_dict(posts, compund_score_dict)
                    pass
                else:
                    compund_score_dict, sorted_topics = self.get_score_dict_and_sorted_posts(claim_words,
                                                                                             irrelevant_words,
                                                                                             msg, posts)
                # if iteration < 2:
                #     self._posts_for_labeling = 50
                # else:
                #     self._posts_for_labeling = 20
                posts_for_labeling = [p for p in sorted_topics if p.post_id not in labeled_posts][
                                     :self._posts_for_labeling]
                labeled_posts.update([p.post_id for p in posts_for_labeling])

                rels = [p for p in posts_for_labeling if self.get_post_rel_to_claim(claim, p, topic_doc_rel) >= 1]
                add_rel = True if (len(rels) == 0) else False
                rel_posts += rels
                irrel_posts += [p for p in posts_for_labeling if
                                self.get_post_rel_to_claim(claim, p, topic_doc_rel) == 0]

                self.add_posts_words_to_set(claim_words, rel_posts)
                self.add_posts_words_to_set(irrelevant_words, irrel_posts)
                print()
                print('Claim id {} {}/ {}, Iteration {}, rel: {}, irrel: {}'.format(claim.claim_id, str(i + 1),
                                                                                    len(claims), iteration,
                                                                                    len(rel_posts), len(irrel_posts)))

                iteration_print = [(claim.claim_id, iteration, len(rel_posts), len(irrel_posts))]
                df = pd.DataFrame(iteration_print, columns=['claim_id', 'iteration', 'rel', 'irrel'])
                file_name = 'iteration_summary_iteration_{}_labeling_{}.csv'.format(self._interactive_iterations,
                                                                                    self._posts_for_labeling)
                output_iteration_file = os.path.join(self._output_path, file_name)
                if os.path.isfile(output_iteration_file):
                    df.to_csv(output_iteration_file, mode='a', header=None)
                else:
                    df.to_csv(output_iteration_file)

                # common_words = claim_words & irrelevant_words
                # claim_words = claim_words - common_words
                # irrelevant_words = irrelevant_words - common_words

                self.mark_labeled_posts(compund_score_dict, irrel_posts, rel_posts)
                rows = self.get_eval_rows_for_posts(claim, compund_score_dict, sorted_topics)
                iteration_rows[iteration] = iteration_rows[iteration] + rows

            print('Summary: rel {}, irrel {}'.format(len(rel_posts), len(irrel_posts)))
            posts = source_id_elements[claim.claim_id]

            if doc_rank_model is not None:
                post_rank = doc_rank_model.rank(' '.join(claim_words))
                compund_score_dict = {p.post_osn_id: rank for p, rank in zip(posts, post_rank)}
                sorted_topics = self._sort_posts_using_post_dict(posts, compund_score_dict)
                pass
            else:
                msg = '\rcompute distance for claim {}/{}'.format(str(i + 1), len(posts))
                compund_score_dict, sorted_topics = self.get_score_dict_and_sorted_posts(claim_words, irrelevant_words,
                                                                                         msg, posts)

            self.mark_labeled_posts(compund_score_dict, irrel_posts, rel_posts)

            # self.write_post_eval_for_claim(claim, irrel_posts, irrelevant_score_dict, post_score_dict, rel_posts,
            #                                posts_for_labeling, topic_doc_rel)
            rows = self.get_eval_rows_for_posts(claim, compund_score_dict, posts)
            t_rows.extend(rows)

        for iter, rows in enumerate(iteration_rows):
            output_path = 'trec2012_eval_interactive_RME_{}_iter_{}_labeling_{}.txt'.format(
                self._relevance_score_method, iter, self._posts_for_labeling)
            pd.DataFrame(rows).to_csv(os.path.join(self._output_path, output_path), sep=' ', index=False,
                                      header=None)

        output_path = 'trec2012_eval_interactive_RME_{}_iter_{}_labeling_{}.txt'.format(
            self._relevance_score_method, len(iteration_rows), self._posts_for_labeling)
        pd.DataFrame(t_rows).to_csv(os.path.join(self._output_path, output_path), sep=' ', index=False,
                                    header=None)
        pass

    def get_score_dict_and_sorted_posts(self, claim_words, irrelevant_words, msg, posts):
        post_score_dict = self._get_posts_score_dict_for_claim(claim_words, posts, msg)
        # irrelevant_score_dict = self._get_posts_score_dict_for_claim(irrelevant_words, posts, msg, 0.0)
        # compund_score_dict = self._get_compund_dict(irrelevant_score_dict, post_score_dict)
        compund_score_dict = post_score_dict
        sorted_posts = self._sort_posts_using_post_dict(posts, compund_score_dict)
        return compund_score_dict, sorted_posts

    def get_eval_rows_for_posts(self, claim, compund_score_dict, sorted_posts):
        rows = []
        for post in sorted_posts:
            rows.append((claim.claim_id, post.post_osn_id, compund_score_dict[post.post_osn_id],
                         'eval_interactive_RME'))
        return rows

    def mark_labeled_posts(self, compund_score_dict, irrel_posts, rel_posts):
        for post in rel_posts:
            compund_score_dict[post.post_osn_id] = 0.0
        for post in irrel_posts:
            compund_score_dict[post.post_osn_id] = 1000.0

    def write_post_eval_for_claim(self, claim, irrel_posts, irrelevant_score_dict, post_score_dict, rel_posts,
                                  sorted_posts, topic_doc_rel):
        rows_for_eval = []
        for post in sorted_posts:
            post_words = clean_claim_description(post.content, True).split()
            rows_for_eval.append((claim.claim_id, post.post_osn_id, post.content, len(post_words),
                                  post_score_dict[post.post_osn_id], irrelevant_score_dict[post.post_osn_id],
                                  topic_doc_rel['{}{}'.format(claim.claim_id, post.post_id)], len(rel_posts),
                                  len(irrel_posts)))
        df = pd.DataFrame(rows_for_eval,
                          columns=['claim_id', 'post_id', 'post_content', 'post_len', 'rel_post_score',
                                   'irrel_post_score', 'label', 'rel_count', 'irrel_count'])
        df.to_excel(os.path.join(self._output_path, 'claim_{}_eval.xlsx'.format(claim.claim_id)))

    def get_post_rel_to_claim(self, claim, p, topic_doc_rel):
        return topic_doc_rel['{}{}'.format(claim.claim_id, p.post_id)]

    def add_posts_words_to_set(self, claim_words, rel_posts):
        list(map(claim_words.update, [clean_claim_description(p.content, True).split() for p in rel_posts]))

    def _get_compund_dict(self, irrelevant_score_dict, post_score_dict):
        return {key: post_score_dict[key] / (1.0 + irrelevant_score_dict[key]) for key in
                post_score_dict}

    def output_data_for_rec_req_algorithm(self):
        table = pd.read_sql_table('posts', self._db.engine)
        table = table[['post_osn_id', 'content']]
        table['content'] = [text.replace('\n', ' ').replace('\t', ' ') for text in table['content']]
        table.to_csv('docs_for_rec_req' + '.txt', sep='\t', header=None, index=False)

        claims = pd.read_sql_table('claims', self._db.engine)
        claims = claims[['claim_id', 'keywords', 'verdict_date']]
        claims['verdict_date'] = [time.mktime(d.timetuple()) for d in claims['verdict_date']]
        claims.to_csv('claims_for_rec' + '.csv', sep=',', header=None, index=False)

    def fill_claim_post_connection_using_keywords(self):
        self.build_word_post_dict_for_trec()
        claim_tweets_connections = []
        claims = self._db.get_claims()
        claim_id_keywords_dict = self._db.get_claim_id_keywords_dict_by_connection_type('hill_climbing')
        for claim in claims:
            claim_keywords_list = claim_id_keywords_dict[claim.claim_id]
            for claim_keywords in claim_keywords_list:
                posts = self.get_posts_from_word_post_dict(claim_keywords, claim)
                claim_tweets_connections += [Claim_Tweet_Connection(claim_id=claim.claim_id, post_id=p.post_id) for p in
                                             posts]
            if len(claim_tweets_connections) > 10000:
                self._db.add_claim_tweet_connections_fast(claim_tweets_connections)
                claim_tweets_connections = []
        self._db.add_claim_tweet_connections_fast(claim_tweets_connections)
        del claim_tweets_connections

    def trec_2102_black_box_eval(self):
        from dataset_builder.ClaimKeywordsFinder import ClaimKeywordFinder
        ckf = ClaimKeywordFinder(self._db)
        ckf.setUp()
        ckf._use_posts_as_corpus = True
        ckf._corpus_domain = 'Trec2012'
        ckf._claim_without_keywords_only = False

        self.build_word_post_dict_for_trec()
        topic_doc_rel = self._get_topic_doc_rel_dict('data/input/trec_data/adhoc-qrels_filtered')
        claims = self._db.get_claims_by_domain('Trec2012')
        args = self.generate_args_for_keywords('Trec2012')
        source_id_elements = self._get_source_id_target_elements(args)
        iteration_rows = [[]] * (self._interactive_iterations + 1)
        t_rows = []
        claim_id_keywords_dict = {c.claim_id: c.keywords.lower() for c in claims}
        for claim in claims:
            claim.description = claim.keywords.lower()
        self._db.addPosts(claims)

        claim_rel_posts_dict = defaultdict(list)
        claim_irrel_posts_dict = defaultdict(list)
        claim_labeled_posts_dict = defaultdict(set)
        claim_irrel_words_dict = defaultdict(set)

        for iteration in range(self._interactive_iterations + 1):
            for i, claim in enumerate(claims):
                # if claim.claim_id not in {'62', '64', '103', '68', '69', '87', '77', '98', '74', '102', '90', '100',
                #                           '101', '95', '107', '104', '105', '99', '59', '54', '57', '56', '109'}:
                #     continue
                claim_words = set(claim.description.split())
                keywords_list = claim_id_keywords_dict[claim.claim_id]
                claim_keywords_list = [keywords.split() for keywords in keywords_list.split('||')]

                # claim.description = claim_words
                # self._db.addPosts([claim])

                claim_labeled_posts = source_id_elements[claim.claim_id]
                posts_with_judgement = set(p.post_id for p in claim_labeled_posts)

                irrelevant_words = claim_irrel_words_dict[claim.claim_id]
                labeled_posts = claim_labeled_posts_dict[claim.claim_id]
                rel_posts = claim_rel_posts_dict[claim.claim_id]
                irrel_posts = claim_irrel_posts_dict[claim.claim_id]

                posts = []
                seen_posts = set()
                for claim_keywords in claim_keywords_list:
                    posts += [p for p in self.get_posts_from_word_post_dict(claim_keywords, claim) if
                              p.post_id not in seen_posts]
                    seen_posts.update([p.post_id for p in posts])
                posts = [p for p in posts if p.post_id in posts_with_judgement]
                msg = '\rcompute distance for claim {}/{}'.format(str(i + 1), len(posts))

                compund_score_dict, sorted_topics = self.get_score_dict_and_sorted_posts(claim_words,
                                                                                         irrelevant_words,
                                                                                         msg, posts)

                posts_for_labeling = [p for p in sorted_topics if p.post_id not in labeled_posts][
                                     :self._posts_for_labeling]
                labeled_posts.update([p.post_id for p in posts_for_labeling])

                rels = [p for p in posts_for_labeling if self.get_post_rel_to_claim(claim, p, topic_doc_rel) >= 1]
                rel_posts += rels
                irrel_posts += [p for p in posts_for_labeling if
                                self.get_post_rel_to_claim(claim, p, topic_doc_rel) == 0]

                self.add_posts_words_to_set(claim_words, rel_posts)
                self.add_posts_words_to_set(irrelevant_words, irrel_posts)
                # claim_words = claim_words - irrelevant_words
                claim.description = ' '.join(claim_words)
                self._db.addPosts([claim])

                print()
                print('Claim id {} {}/ {}, Iteration {}, rel: {}, irrel: {}'.format(claim.claim_id, str(i + 1),
                                                                                    len(claims), iteration,
                                                                                    len(rel_posts), len(irrel_posts)))
                print()
                self.mark_labeled_posts(compund_score_dict, irrel_posts, rel_posts)
                rows = self.get_eval_rows_for_posts(claim, compund_score_dict, sorted_topics)
                iteration_rows[iteration] = iteration_rows[iteration] + rows

                claim_labeled_posts_dict[claim.claim_id] = labeled_posts
                claim_rel_posts_dict[claim.claim_id] = rel_posts
                claim_irrel_posts_dict[claim.claim_id] = irrel_posts

            ckf.hill_climbing()
            claim_id_keywords_dict = self._db.get_claim_id_keywords_dict_by_connection_type('hill_climbing_final')

        for iter, rows in enumerate(iteration_rows):
            output_path = 'trec2012_eval_black_box_{}_iter_{}_labeling_{}.txt'.format(
                self._relevance_score_method, iter, self._posts_for_labeling)
            pd.DataFrame(rows).to_csv(os.path.join(self._output_path, output_path), sep=' ', index=False,
                                      header=None)
