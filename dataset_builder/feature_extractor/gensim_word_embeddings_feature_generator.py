import pandas as pd

from dataset_builder.feature_extractor.word_embeddings_feature_generator import WordEmbeddingsFeatureGenerator


class GensimWordEmbeddingsFeatureGenerator(WordEmbeddingsFeatureGenerator):
    def __init__(self, db, **kwargs):
        super(GensimWordEmbeddingsFeatureGenerator, self).__init__(db, **kwargs)
        self._author_word_embedding_csv_path = self._config_parser.eval(self.__class__.__name__,
                                                                        "author_word_embedding_csv_path")

    def load_author_guid_word_embedding_dict(self, targeted_field_name, targeted_table, targeted_word_embedding_type):
        author_word_embedding_df = pd.DataFrame.from_csv(self._author_word_embedding_csv_path)
        df = author_word_embedding_df[(author_word_embedding_df['table_name'] == targeted_table) &
                                      (author_word_embedding_df['targeted_field_name'] == targeted_field_name) &
                                      (author_word_embedding_df[
                                           'word_embedding_type'] == targeted_word_embedding_type)]
        author_guid_word_embeding_dict = self._db.create_author_guid_word_embedding_dict_by_recoreds(df.values.tolist())
        return author_guid_word_embeding_dict