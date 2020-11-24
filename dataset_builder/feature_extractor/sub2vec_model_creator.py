
import os
from functools import partial

import gensim.models.doc2vec as doc
import networkx as nx
import pandas as pd
from sklearn.utils.multiclass import unique_labels

from dataset_builder.Sub2VecCode import structural
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from dataset_builder.word_embedding.abstract_word_embadding_trainer import AbstractWordEmbaddingTrainer


class Sub2VecModelCreator(AbstractWordEmbaddingTrainer):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **{'authors': [], 'posts': {}})
        self._walks_count = self._config_parser.eval(self.__class__.__name__, "walks_count")
        self._walk_length = self._config_parser.eval(self.__class__.__name__, "walk_length")
        self._window_size = self._config_parser.eval(self.__class__.__name__, "window_size")
        self._epochs = self._config_parser.eval(self.__class__.__name__, "epochs")
        self._num_of_dimensions = self._config_parser.eval(self.__class__.__name__, "num_of_dimensions")
        self._table_name = self._config_parser.eval(self.__class__.__name__, "table_name")

    def execute(self, window_start=None):
        graphs = self.load_graphs()
        dimensions = self._num_of_dimensions
        args = {'dimensions': dimensions,
                'window': self._window_size,
                'walkLength': self._walk_length,
                'iterations': self._epochs,
                'alpha': 0.8,
                'dm': 0,
                'randomWalkCount': self._walks_count}
        embeddings = self.graph_structural_embedding(graphs, **args)
        photo_embedding_rows = []
        post_osn_to_id = {post.post_osn_id: post.post_id for post in self._db.get_posts()}
        for graph, embedding in zip(graphs, embeddings):
            photo_embedding_rows.append([post_osn_to_id.get(graph.graph['name'])] + list(embedding))
        columns = ['photo_id'] + list(map(str, list(range(self._num_of_dimensions))))
        print('save data to DB')
        photo_embeddings_df = pd.DataFrame(photo_embedding_rows, columns=columns)
        photo_embeddings_df.to_sql(name=self._table_name, con=self._db.engine, index=False, if_exists='replace')

    def load_graphs(self):
        posts = self._db.get_posts()
        author_connections_dict = self._db.get_author_connections_dict()
        authors_by_domain_dict = self._db.get_authors_by_domain_dict()
        graphs = []
        for i, post in enumerate(posts):
            print('\r create graph {}/{}'.format(str(i+1), len(posts)), end='')
            if post.post_type is not None:
                graph_id = post.post_osn_id
                connections = author_connections_dict.get(str(graph_id), [])
                edges = [(connection.source_author_guid, connection.destination_author_guid) for connection in
                         connections]
                nodes = [author.author_guid for author in authors_by_domain_dict.get(str(graph_id), [])]
                graph = nx.DiGraph()
                graph.add_nodes_from(nodes)
                graph.add_edges_from(edges)
                graphs.append(graph)
                graph.graph['name'] = graph_id
                graph.graph['tags'] = unique_labels(post.post_type.split(','))
        print()
        return graphs

    def graph_structural_embedding(self, graphs, **kwargs):
        dirName = 'data/output/sub2vec_output/'
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        file_name = os.path.join(dirName, 'random_walk_file.walk')
        indexToName = structural.generateWalkFile(graphs, file_name, kwargs['walkLength'], kwargs['alpha'],
                                                  kwargs['randomWalkCount'])
        sentences = doc.TaggedLineDocument(file_name)
        print('build model')
        model = doc.Doc2Vec(sentences, vector_size=kwargs['dimensions'], epochs=kwargs['iterations'], dm=kwargs['dm'],
                            window=kwargs['window'])

        # outputfile = os.path.join(dirName, 'vectors.vec')
        # print('save vectores')
        # structural.saveVectors(model.docvecs, outputfile, indexToName)
        return model.docvecs

    def convert_embeddings_to_features(self, embedding, graph):
        author_features = []
        attr_template = '{}_dim{}'
        guid = graph.graph['name']
        prefix = self.__class__.__name__
        create_feature = partial(self.create_author_feature, author_guid=guid, window_start=None, window_end=None)
        for i, val in enumerate(embedding):
            feature = create_feature(feature_name=attr_template.format(prefix, i), attribute_value=val)
            author_features.append(feature)
        return author_features

    def convert_embedding_to_author_features(self, graphs, embeddings):
        author_features = []
        for graph, embedding in zip(graphs, embeddings):
            author_features += self.convert_embeddings_to_features(embedding, graph)
        return author_features
