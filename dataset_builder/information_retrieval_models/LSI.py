import string
from typing import List, Type, Tuple, Union
from abc import ABCMeta, abstractmethod
import numpy as np
from tqdm.auto import tqdm
from commons.commons import clean_claim_description
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_term_count_model(docs):
    return build_term_score_from_vectorizer(docs, CountVectorizer())


def build_term_tf_idf_model(docs):
    return build_term_score_from_vectorizer(docs, TfidfVectorizer())


def build_term_score_from_vectorizer(docs, vectorizer):
    docs_str = [' '.join(d) for d in docs]
    tf_idf_vectoraizer = vectorizer
    model = tf_idf_vectoraizer.fit_transform(docs_str).T
    words = tf_idf_vectoraizer.get_feature_names()
    return model.toarray(), sorted(words)


class LSI_m:
    """Latent Semantic Indexing.
        """
    def __init__(self, docs, model = build_term_count_model, rank_approximation = 2, stopwords = None,
                 ignore_chars=string.punctuation):

        if stopwords is None:
            stopwords = []
        self.stopwords = stopwords
        self.ignore_chars = ignore_chars
        self.docs = []
        for doc in tqdm(docs, desc='LSI process docs', total=len(docs)):
            # assert isinstance(doc, str)
            self.docs.append(clean_claim_description(doc, True).split())

        # self.words = self._get_words(self.docs)
        # self.query = self._parse_query(query)
        self.rank_approximation = rank_approximation
        self.term_doc_matrix, self.words = model(self.docs)
        self._svd_components = self._svd_with_dimensionality_reduction(self.term_doc_matrix, self.rank_approximation)

    def _get_words(self, docs_words):
        words = set()
        for doc_words in docs_words:
            words = words | set(doc_words)

        return sorted(words)


    def _svd_with_dimensionality_reduction(self, term_doc_matrix, rank_approximation):
        u, s, v = np.linalg.svd(term_doc_matrix)
        s = np.diag(s)
        k = rank_approximation
        return u[:, :k], s[:k, :k], v[:, :k]

    def rank(self, query):
        # assert isinstance(query, str)
        query_words = self._parse_query(query)
        u_k, s_k, v_k = self._svd_components

        q = np.matmul(np.matmul(query_words.T, u_k), np.linalg.pinv(s_k))
        d = np.matmul(np.matmul(self.term_doc_matrix.T, u_k), np.linalg.pinv(s_k))

        res = np.apply_along_axis(lambda row: self._sim(q, row), axis=1, arr=d)
        ranking = np.argsort(-res) + 1
        return ranking

    def _parse_query(self, query):
        result = np.zeros(len(self.words))

        i = 0
        for word in sorted(clean_claim_description(query, True).split()):
            while word > self.words[i]:
                i += 1
            if word == self.words[i]:
                result[i] += 1

        return result

    @staticmethod
    def _sim(x, y):
        return np.matmul(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# class LSI:
#     """Latent Semantic Indexing.
#     """
#
#     def __init__(self, docs, query, model = TermCountModel,
#                  rank_approximation = 2, stopwords = None,
#                  ignore_chars=string.punctuation):
#         if stopwords is None:
#             stopwords = []
#         self.stopwords = stopwords
#         self.ignore_chars = ignore_chars
#         self.docs = list(map(self._parse, docs))
#         self.words = self._get_words()
#         self.query = self._parse_query(query)
#         self.model = model
#         self.rank_approximation = rank_approximation
#         self.term_doc_matrix = self._build_term_doc_matrix()
#
#     def _parse(self, text):
#         translator = string.maketrans(self.ignore_chars, ' ' * len(self.ignore_chars))
#         return list(map(str.lower,
#                         filter(lambda w: w not in self.stopwords,
#                                text.translate(translator).split())))
#
#     def _parse_query(self, query):
#         result = np.zeros(len(self.words))
#
#         i = 0
#         for word in sorted(self._parse(query)):
#             while word > self.words[i]:
#                 i += 1
#             if word == self.words[i]:
#                 result[i] += 1
#
#         return result
#
#     def _get_words(self):
#         words = set()
#
#         for doc in self.docs:
#             words = words | set(doc)
#
#         return sorted(words)
#
#     def _build_term_doc_matrix(self):
#         model = self.model(self.words, self.docs)
#         return model.build()
#
#     def _svd_with_dimensionality_reduction(self):
#         u, s, v = np.linalg.svd(self.term_doc_matrix)
#         s = np.diag(s)
#         k = self.rank_approximation
#         return u[:, :k], s[:k, :k], v[:, :k]
#
#     def process(self):
#         u_k, s_k, v_k = self._svd_with_dimensionality_reduction()
#
#         q = np.matmul(self.query.T, u_k @ np.linalg.pinv(s_k))
#         d = np.matmul(np.matmul(self.term_doc_matrix.T, u_k), np.linalg.pinv(s_k))
#
#         res = np.apply_along_axis(lambda row: self._sim(q, row), axis=1, arr=d)
#         ranking = np.argsort(-res) + 1
#         return ranking
#
#     @staticmethod
#     def _sim(x, y):
#         return np.matmul(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


