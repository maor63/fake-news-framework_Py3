# Lior Bass 26.10.17

from dataset_builder.feature_extractor.word_embeddings_feature_generator import WordEmbeddingsFeatureGenerator


class GloveWordEmbeddingsFeatureGenerator(WordEmbeddingsFeatureGenerator):
    def load_author_guid_word_embedding_dict(self, targeted_field_name, targeted_table,
                                             targeted_word_embedding_type):
        author_guid_word_embeding_dict = self._db.get_author_guid_word_embedding_vector_dict(self._word_embedding_table_name,
                                                                                             targeted_table,
                                                                                            targeted_field_name,
                                                                                            targeted_word_embedding_type)
        return author_guid_word_embeding_dict


