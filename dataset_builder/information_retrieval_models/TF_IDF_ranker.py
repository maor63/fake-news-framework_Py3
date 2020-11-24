from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
from commons.commons import clean_claim_description
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TF_IDF_Ranker:
    def __init__(self, docs):
        self.docs = []
        for doc in tqdm(docs, desc='TF-IDF process docs', total=len(docs)):
            self.docs.append(clean_claim_description(doc, True).split())

        docs_str = [' '.join(d) for d in self.docs]
        self.tf_idf_vectoraizer = TfidfVectorizer(stop_words='english',)
        self.doc_vectors = self.tf_idf_vectoraizer.fit_transform(docs_str).toarray()

    def rank(self, query):
        query_text = clean_claim_description(query, True)
        query_vector = self.tf_idf_vectoraizer.transform([query_text]).toarray()
        distances = cdist(query_vector, self.doc_vectors, metric='cosine')[0]
        return distances