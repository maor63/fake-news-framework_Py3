from tqdm.auto import tqdm
from commons.commons import clean_claim_description
from gensim.summarization.bm25 import BM25
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class BM25Model:
    """Latent Semantic Indexing.
        """
    def __init__(self, docs):
        self.docs = []
        for doc in tqdm(docs, desc='LSI process docs', total=len(docs)):
            self.docs.append(clean_claim_description(doc, True).split())
        self.bm25_model = BM25(self.docs)

        docs_str = [' '.join(d) for d in self.docs]
        tf_idf_vectoraizer = TfidfVectorizer(stop_words='english',)
        tf_idf_vectoraizer.fit(docs_str)
        self.avg_idf = tf_idf_vectoraizer.idf_.mean()

    def rank(self, query):
        query_words = clean_claim_description(query, True).split()
        return -np.array(self.bm25_model.get_scores(query_words, self.avg_idf))