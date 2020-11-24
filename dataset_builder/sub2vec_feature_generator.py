from dataset_builder.feature_extractor.word_embeddings_feature_generator import WordEmbeddingsFeatureGenerator
import pandas as pd


class Sub2VecFeatureGenerator(WordEmbeddingsFeatureGenerator):
    def load_author_guid_word_embedding_dict(self, targeted_field_name, targeted_table, targeted_word_embedding_type):
        embedding_df = pd.read_sql_table(self._word_embedding_table_name, self._db.engine)
        return {embedding[0]: embedding[1:] for embedding in embedding_df.values}
