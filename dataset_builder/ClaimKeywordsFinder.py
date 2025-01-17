
import random
import timeit
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
import numpy as np
from sortedcollections import OrderedSet

from DB.schema_definition import Claim_Keywords_Connections
from commons.commons import *
from commons.method_executor import Method_Executor
from dataset_builder.keywords_evaluator import KeywordEvaluator
from old_tweets_crawler.old_tweets_crawler import OldTweetsCrawler
from preprocessing_tools.keywords_generator import KeywordsGenerator
from nltk import pos_tag
import pandas as pd
from nltk.tokenize import word_tokenize


class ClaimKeywordFinder(Method_Executor):

    def __init__(self, db):
        super(ClaimKeywordFinder, self).__init__(db)

        self._max_keywords_size = self._config_parser.eval(self.__class__.__name__, "max_keywords_size")
        self._min_keywords_size = self._config_parser.eval(self.__class__.__name__, "min_keywords_size")
        self._output_keywords_count = self._config_parser.eval(self.__class__.__name__, "output_keywords_count")
        self._keywords_start_size = self._config_parser.eval(self.__class__.__name__, "keywords_start_size")
        self._iteration_count = self._config_parser.eval(self.__class__.__name__, "iteration_count")
        self._start_from_claim = self._config_parser.eval(self.__class__.__name__, "start_from_claim")
        self._use_posts_as_corpus = self._config_parser.eval(self.__class__.__name__, "use_posts_as_corpus")
        self._claim_without_keywords_only = self._config_parser.eval(self.__class__.__name__, "claim_without_keywords_only")
        self._corpus_domain = self._config_parser.eval(self.__class__.__name__, "corpus_domain")
        self._search_count = self._config_parser.eval(self.__class__.__name__, "search_count")  # same search count
        self._word_post_dictionary = defaultdict(set)
        self._post_dictionary = {}
        self._exploration_probability = 1.0
        self._decay = 0.85
        self._num_of_walks = 3
        self._min_tweet_count = self._config_parser.eval(self.__class__.__name__, "min_tweet_count")
        self._min_distance = 3.7
        self._tweet_crawler = OldTweetsCrawler(db)
        self._tweet_crawler._max_num_tweets = 20
        # self._keywords_generator = KeywordsGenerator(db)
        self._keywords_evaluator = KeywordEvaluator(db)
        self._keywords_connections = []
        self._walked_keywords = Counter()
        self._last_distance = 1000
        self._keywords_score_dict = defaultdict()

    def setUp(self):
        self._keywords_evaluator.setUp()
        if self._use_posts_as_corpus:
            self._word_post_dictionary = defaultdict(set)
            print("Load posts dictionary")
            self._post_dictionary = self._db.get_post_dictionary()
            posts = self._db.get_posts_filtered_by_domain(self._corpus_domain)
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

    def bottom_up_search(self):
        self._greedy_search(self._get_next_bottom_up_keywords, 'bottom_up')

    def _greedy_search(self, generate_next_keywords, search_type):
        claims = self._db.get_claims()
        min_tweets = self._min_tweet_count
        # word_tf_idf_dict = self._keywords_generator.get_word_tf_idf_of_claims()
        for i, claim in enumerate(claims):
            keywords_tweet_num = defaultdict(int)
            if i < self._start_from_claim:
                continue
            walked_keywords = Counter()
            print('{} Claim {}/{}'.format(search_type, i, len(claims)))
            start = timeit.default_timer()
            claim_description_words = self.get_claim_words_from_description(claim)
            ordered_words = OrderedSet(claim_description_words)
            base_keywords = ordered_words if search_type == 'top_down' else set()
            # num_of_potential_words = len(ordered_words)
            num_of_potential_words = self._max_keywords_size
            keywords_list = []
            for size in range(1, num_of_potential_words + 1):
                same_keywords_size = Counter()
                best_word_rank_tuple = ['', 1000]
                for iter, word in enumerate(ordered_words):
                    keywords_str = ' '.join(generate_next_keywords(base_keywords, word))
                    keywords_size = len(keywords_str.split(' '))
                    type_name = '{}_iter_{}_keywords_size_{}'.format(search_type, iter, keywords_size)
                    evaluation = self.eval_keywords_for_claim(claim, keywords_str, type_name)
                    if best_word_rank_tuple[1] > evaluation['distance'] and evaluation['tweet_num'] > min_tweets:
                        best_word_rank_tuple = [word, evaluation['distance']]
                    print('\r{} Distance: {}'.format(type_name, evaluation['distance']), end='')
                    keywords_tweet_num[keywords_str] = evaluation['tweet_num']
                    if evaluation['tweet_num'] > min_tweets:
                        walked_keywords[keywords_str] = -1.0 * evaluation['distance']
                        same_keywords_size[keywords_str] = -1.0 * evaluation['distance']
                    keywords_list.append([keywords_str, evaluation['distance'], type_name])
                if ordered_words:
                    if best_word_rank_tuple[0] == '':
                        best_word_rank_tuple[0] = ordered_words.pop()
                    else:
                        ordered_words.discard(best_word_rank_tuple[0])
                base_keywords = generate_next_keywords(base_keywords, best_word_rank_tuple[0])

                curr_distance = best_word_rank_tuple[1]
                if len(same_keywords_size) > 0:
                    keywords, best_distances = same_keywords_size.most_common(1)[0]
                    self._add_new_keywords(claim, keywords,
                                           '{}_keywords_size_{}'.format(search_type, size), -1.0 * best_distances,
                                           keywords_tweet_num[keywords])
                else:
                    self._add_new_keywords(claim, ' '.join(base_keywords),
                                           '{}_keywords_size_{}'.format(search_type, size), curr_distance,
                                           keywords_tweet_num[' '.join(base_keywords)])

            for keywords, keywords_distance, type_name in keywords_list:
                self._add_new_keywords(claim, keywords, type_name, keywords_distance, keywords_tweet_num[keywords])

            if len(walked_keywords) > 0:
                keywords, best_distances = list(zip(*(walked_keywords.most_common(self._output_keywords_count))))
            else:
                sorted_by_second = sorted(keywords_list, key=lambda tup: tup[1], reverse=True)
                keywords, best_distances, type_name = list(zip(*sorted_by_second[:self._output_keywords_count]))

            self._add_new_keywords(claim, '||'.join(keywords), '{}_final'.format(search_type),
                                   -1.0 * np.mean(best_distances),
                                   sum(list(keywords_tweet_num[k] for k in keywords)))
            with self._db.session.no_autoflush:
                self._db.addPosts(self._keywords_connections)
            self._keywords_connections = []
            end = timeit.default_timer()
            print('run time: {}'.format((end - start)))
        # self._db.session.expire_on_commit = True

    def get_synonms_for_words(self, words, limit=5):
        synonyms = []
        for word in set(words):
            word_syns = []
            for syn in wordnet.synsets(word):
                if len(word_syns) > limit:
                    break
                for l in syn.lemmas():
                    word_syns.append(l.name())
            synonyms += word_syns
        return list(set(synonyms) - set(words))

    def get_near_neighbors_for_words(self, words, limit=5):
        similar_words = []
        for word in words:
            ms = self._keywords_evaluator._model.most_similar(word, topn=limit)
            for w, similarity in ms:
                similar_words.append(w)
        return list(set(similar_words) - set(words))

    def get_claim_words_from_description(self, claim):
        # return clean_claim_description(u' '.join(self.get_name_entities(text)), True).split(' ')
        words = self.get_name_entities(claim.title.rstrip('.').split()) + self.get_name_entities(
            claim.description.rstrip('.').split())

        # words = words + self.get_synonms_for_words(words)
        # words = words + self.get_near_neighbors_for_words(words)
        return words

    def _get_next_top_down_keywords(self, base_keywords, word):
        return base_keywords - {word}

    def _get_next_bottom_up_keywords(self, base_keywords, word):
        return base_keywords.union({word})

    def simulated_anniling(self):
        search_type = 'simulated_anniling'
        iteration_count = self._iteration_count
        exploration_probability = 1.0
        keywords_start_size = self._keywords_start_size
        decay = 0.85
        self._base_stochastic_search(iteration_count, keywords_start_size, exploration_probability, search_type, decay)

    def hill_climbing(self):
        search_type = 'hill_climbing'
        iteration_count = self._iteration_count
        exploration_probability = 0.0
        keywords_start_size = self._keywords_start_size
        decay = 1.0
        self._base_stochastic_search(iteration_count, keywords_start_size, exploration_probability, search_type, decay)

    def random_walk(self):
        search_type = 'random_walk'
        iteration_count = self._iteration_count
        exploration_probability = 1.0
        keywords_start_size = self._keywords_start_size
        decay = 1.0
        self._base_stochastic_search(iteration_count, keywords_start_size, exploration_probability, search_type, decay)

    def _base_stochastic_search(self, iterations, keywords_start_size, exploration, search_type, decay):
        self._decay = decay
        if self._claim_without_keywords_only:
            claims = self._db.get_claims_without_keywords()
        else:
            claims = self._db.get_claims()
        for i, claim in enumerate(claims):
            # if claim.claim_id not in {'62', '64', '103', '68', '69', '87', '77', '98', '74', '102', '90', '100',
            #                           '101', '95', '107', '104', '105', '99', '59', '54', '57', '56', '109'}:
            #     continue
            if i < self._start_from_claim:
                continue
            final_queries = []
            self._keywords_score_dict = defaultdict()
            start = timeit.default_timer()
            for j in range(self._search_count):
                all_keywords = set()

                self._walked_keywords = Counter()
                self._last_distance = 1000
                keywords_by_size = defaultdict(list)
                self._exploration_probability = exploration
                prune_set = set()
                print('{} Claim {}/{} search {}'.format(search_type, i, len(claims), j))

                claim_description_words = self.get_claim_words_from_description(claim)
                claim_description_words = list([x for x in claim_description_words if x != ''])
                # new_keywords = self._get_random_keywords_for_claim(claim_description_words, keywords_start_size)
                new_keywords = self._get_keywords_by_pos_tagging_for_claim(claim_description_words, keywords_start_size)
                # new_keywords = str.lower(str(claim.keywords)).split()
                current_keywords = new_keywords
                iteration = 0
                while iteration < iterations:
                    try:
                        type_name = '{}_iter_{}_keywords_size_{}_search_{}'.format(search_type, iteration,
                                                                                    len(new_keywords), j)
                        keywords_by_size[len(new_keywords)].append(new_keywords)
                        print('\r' + type_name + ' exploration: {}'.format(self._exploration_probability), end='')

                        keywords_set = ' '.join(new_keywords)
                        all_keywords.add(keywords_set)
                        evaluation = self.eval_keywords_for_claim(claim, keywords_set, type_name)
                        self._add_new_keywords(claim, keywords_set, type_name, evaluation['distance'],
                                               evaluation['tweet_num'])
                        if evaluation['tweet_num'] == 0:
                            prune_set.add(frozenset(new_keywords))

                        if self._evaluate_keywords_for_simulated(evaluation, new_keywords):
                            current_keywords = new_keywords

                        new_keywords = self._get_next_keywords(claim_description_words, current_keywords, prune_set)

                        iteration += 1
                        self._exploration_probability *= decay
                        if new_keywords is None:
                            break
                    except Exception as e:
                        print(e)
                        break

                self._add_keywords_by_size(claim, j, keywords_by_size, search_type)

                queries = []
                if len(self._walked_keywords) > 0:
                    queries, best_distances = list(zip(*self._walked_keywords.most_common(self._output_keywords_count)))
                    queries = list(queries)
                if self._output_keywords_count - len(queries) > 0:
                    # all_keywords = self._keywords_score_dict.keys()
                    queries += sorted(all_keywords, key=lambda x: self._keywords_score_dict[x][0])[
                               :(self._output_keywords_count - len(queries))]

                # self._add_new_keywords(claim, keywords_str, keywords_type, *self._keywords_score_dict[keywords_str])
                distances, tweet_counts = list(zip(
                    *[self._keywords_score_dict[keywords_set] for keywords_set in queries]))
                self._add_new_keywords(claim, '||'.join(queries), '{}_final_search_{}'.format(search_type, j),
                                       np.mean(distances),
                                       sum(tweet_counts))

                final_queries.extend(queries)

            keywords_set_score_dict = {frozenset(keywords.split()): self._keywords_score_dict[keywords] for keywords in
                                       self._keywords_score_dict}
            final_queries = self._get_sorted_queries(final_queries, keywords_set_score_dict)
            min_count_queries = []
            other_queries = []
            empty_queries = []
            final_queries = sorted(final_queries, key=lambda q: keywords_set_score_dict[q][1], reverse=True)
            for query in final_queries:
                if keywords_set_score_dict[query][1] >= self._min_tweet_count:
                    min_count_queries.append(query)
                elif keywords_set_score_dict[query][1] > 0:
                    other_queries.append(query)
                else:
                    empty_queries.append(query)


            final_queries = min_count_queries + other_queries
            final_queries = final_queries if len(final_queries) > 0 else empty_queries
            distances, tweet_counts = list(zip(*[keywords_set_score_dict[keywords_set] for keywords_set in final_queries]))

            final_queries = [' '.join(query) for query in final_queries][:self._output_keywords_count]
            self._add_new_keywords(claim, '||'.join(final_queries), '{}_final'.format(search_type),
                                   np.mean(distances),
                                   sum(tweet_counts))

            with self._db.session.no_autoflush:
                self._db.addPosts(self._keywords_connections)

            # self.save_keywords_to_csv('claim_keyword_connection_search.csv')

            self._keywords_connections = []
            end = timeit.default_timer()
            print('run time: {}'.format((end - start)))

    def save_keywords_to_csv(self, claim_connection_csv_path):
        keywords_connections_tuples = [connection.__dict__ for connection in self._keywords_connections]
        new_keywords = pd.DataFrame(keywords_connections_tuples)
        if os.path.isfile(claim_connection_csv_path):
            old_keywords = pd.read_csv(claim_connection_csv_path)
            new_keywords = pd.concat([old_keywords, new_keywords])
        new_keywords.to_csv('%s' % claim_connection_csv_path)

    def _get_sorted_queries(self, final_queries, keywords_set_score_dict):
        final_queries = set([frozenset(query.split()) for query in final_queries])  # remove duplicate queries
        final_queries = sorted(final_queries, key=lambda x: keywords_set_score_dict[x][0])  # sort queries by RME
        return final_queries

    def _add_keywords_by_size(self, claim, j, keywords_by_size, search_type):
        for size, keywords_list in keywords_by_size.items():
            keywords = sorted(keywords_list, key=lambda x: self._keywords_score_dict[' '.join(x)][0])[0]
            keywords_type = '{}_keywords_size_{}_search_{}'.format(search_type, size, j)
            keywords_str = ' '.join(keywords)
            self._add_new_keywords(claim, keywords_str, keywords_type, *self._keywords_score_dict[keywords_str])

    def _get_random_keywords_for_claim(self, claim_description_words, keywords_start_size):
        start_size = min(len(claim_description_words), keywords_start_size)
        new_keywords = random.sample(claim_description_words, start_size)
        return new_keywords

    def _get_keywords_by_pos_tagging_for_claim(self, claim_description_words, start_size):
        return self.get_name_entities(claim_description_words)[:start_size]

    def get_name_entities(self, claim_description_words):
        '''
        Get name entities from text using ner tagger from nltk
        :param claim_init_query: input text
        :return: list of named entities
        '''
        # sentence = nltk.word_tokenize(claim_init_query)
        tagging = nltk.pos_tag(claim_description_words)
        namedEnt = nltk.ne_chunk(tagging, binary=True)
        entities = []
        for chunk in namedEnt:
            if hasattr(chunk, 'label'):
                word = ' '.join(c for c, tag in chunk)
                entities.append(remove_punctuation_chars(word.lower()))
            else:
                word, tag = chunk
                if tag in ['VBG', 'CD', 'JJ', 'VB', 'VBN'] or 'NN' in tag:
                    entities.append(remove_punctuation_chars(word.lower()))
        return entities

    def _get_word_to_prob_by_pos_tagging(self, claim_description_words):
        word_pos_tagging_rank_dict = defaultdict(float)
        pos_to_rank = {}
        pos_to_rank['NOUN'] = 5
        pos_to_rank['ADJ'] = 4
        pos_to_rank['ADV'] = 3
        pos_to_rank['NUM'] = 2
        # description = claim.description
        word_tag_tuples = pos_tag(claim_description_words, tagset='universal')
        for word, tag in word_tag_tuples:
            word_pos_tagging_rank_dict[word.lower()] = pos_to_rank.get(tag, 1)
        total = float(sum(word_pos_tagging_rank_dict.values()))
        word_pos_tagging_rank_dict = {word: rank / total for word, rank in word_pos_tagging_rank_dict.items()}
        return word_pos_tagging_rank_dict

    def _evaluate_keywords_for_simulated(self, evaluation, curr_keywords):
        keywords_str = ' '.join(curr_keywords)
        if evaluation['tweet_num'] >= self._min_tweet_count:
            self._walked_keywords[keywords_str] = -1.0 * evaluation['distance']
        if evaluation['distance'] < self._last_distance and evaluation['tweet_num'] > self._min_tweet_count:
            self._last_distance = evaluation['distance']
            return True
        elif random.random() < self._exploration_probability:
            self._last_distance = evaluation['distance']
            return True
        return False

    def _get_next_keywords(self, claim_description_words, current_keywords, prune_set):
        candidates = list(set(claim_description_words) - set(current_keywords))
        next_keywords = self.generate_next_keywords(candidates, current_keywords)
        tries = 0
        while any([prune.issubset(frozenset(next_keywords)) for prune in prune_set]) \
                or ' '.join(next_keywords) in self._keywords_score_dict or \
                len(next_keywords) > self._max_keywords_size:
            try:
                tries += 1
                if tries == 500:
                    return None
                candidates = list(set(claim_description_words) - set(next_keywords))
                next_keywords = self.generate_next_keywords(candidates, next_keywords)
            except Exception:
                pass
        return next_keywords

    def generate_next_keywords(self, possible_candidates, start_position):
        if len(possible_candidates) == 0:
            next_keywords = self.remove_word(start_position)
        elif len(start_position) == self._min_keywords_size:
            prob = random.random()
            if prob < 0.5:
                next_keywords = self.swap_words(possible_candidates, start_position)
            else:
                next_keywords = self.add_word(possible_candidates, start_position)

        else:
            prob = random.random()
            if prob < 0.33:
                next_keywords = self.add_word(possible_candidates, start_position)
            elif prob < 0.67:
                next_keywords = self.swap_words(possible_candidates, start_position)
            else:
                next_keywords = self.remove_word(start_position)
        return next_keywords

    def swap_words(self, possible_candidates, start_position):
        next_keywords = self.remove_word(start_position)
        possible_candidates = list(set(possible_candidates) - set(start_position))
        next_keywords = self.add_word(possible_candidates, next_keywords)
        return next_keywords

    def add_word(self, possible_candidates, start_position):
        word_pos_tagging_rank_dict = self._get_word_to_prob_by_pos_tagging(possible_candidates)
        candidates, probabilities = list(zip(*iter(word_pos_tagging_rank_dict.items())))
        # assert len(set(possible_candidates) & set(start_position)) == 0
        return start_position + list(np.random.choice(candidates, 1, p=probabilities))

    def remove_word(self, start_position):
        word_pos_tagging_rank_dict = self._get_word_to_prob_by_pos_tagging(start_position)
        candidates, probabilities = list(zip(*iter(word_pos_tagging_rank_dict.items())))
        # return random.sample(start_position, len(start_position) - 1)
        return list(np.random.choice(candidates, len(start_position) - 1, False, p=probabilities))

    def eval_keywords_for_claim(self, claim, keywords_str, type_name):
        if self._use_posts_as_corpus:
            # posts = self._db.get_posts_from_domain_contain_words(self._corpus_domain, keywords_str.split())
            posts = self.get_posts_from_word_post_dict(keywords_str.split(), claim)
            evaluation = self._keywords_evaluator.eval_claim_tweets(claim.description, keywords_str, posts)
        else:
            tweets = self._tweet_crawler.retrieve_tweets_by_claim_keywords(claim, keywords_str)
            posts, connections = self._tweet_crawler._convert_tweets_to_posts(tweets, claim.claim_id, '')
            evaluation = self._keywords_evaluator.eval_claim_tweets(claim.description, keywords_str, posts)
        return evaluation

    def _add_new_keywords(self, claim, keywords_str, type_name, score=None, tweet_count=None):
        claim_keywords_connections = Claim_Keywords_Connections()
        claim_keywords_connections.claim_id = claim.claim_id
        claim_keywords_connections.keywords = keywords_str
        claim_keywords_connections.type = type_name
        # if score:
        claim_keywords_connections.score = score
        claim_keywords_connections.tweet_count = tweet_count
        self._keywords_connections.append(claim_keywords_connections)
        self._keywords_score_dict[keywords_str] = (score, tweet_count)
