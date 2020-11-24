import unittest

from DB.schema_definition import *
from dataset_builder.feature_extractor.glove_word_embeddings_feature_generator import \
    GloveWordEmbeddingsFeatureGenerator
from dataset_builder.feature_extractor.gensim_word_embeddings_feature_generator import \
    GensimWordEmbeddingsFeatureGenerator
from dataset_builder.word_embedding.gensim_word_embedding_trainer import GensimWordEmbeddingsModelTrainer
from dataset_builder.word_embedding.glove_word_embedding_model_creator import GloveWordEmbeddingModelCreator


class GloveWordEmbeddingsFeatureGeneratorUnittests(unittest.TestCase):
    def setUp(self):
        self._config_parser = getConfig()
        self._db = DB()
        self._db.setUp()

        self._is_load_wikipedia_300d_glove_model = True
        self._wikipedia_model_file_path = "data/input/glove/test_glove.6B.300d.txt"
        self._table_name = "wikipedia_model_300d"
        self._word_vector_dict_full_path = "data/output/word_embedding/"
        self._word_vector_dict = {}

        self._embedding_table_name = ''
        self._author = None
        self._set_author('test_user')
        self._posts = []

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    def test_0_dimension_of_word(self):
        self._add_post('Post1', 'was')
        self._setup_test()
        db_val = self._get_word_dimension('was', 0)
        self.assertEqual(db_val, 0.065573)

    def test_single_word_single_dimension_feature_correctness(self):
        self._add_post('Post1', 'was')
        self._setup_test()
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_mean_posts_content_{}_d0".format(self._embedding_table_name)).attribute_value
        self.assertAlmostEqual(float(db_val), 0.065573, places=4)

    def test_single_word_single_dimension_feature_correctness_using_gensim(self):
        self._add_post('Post1', 'Thus we saw how we can easily code')
        self._add_post('Post2', 'Thanks for reading this article')
        self._add_post('Post3', 'hello was')
        self._add_post('Post4', 'hello was')
        self._add_post('Post5', 'hello was')
        self._db.session.commit()
        self._word_embedding_model_creator = GensimWordEmbeddingsModelTrainer(self._db)
        self._word_embedding_model_creator.execute(None)

        params = {'authors': [self._author], 'posts': self._posts}
        self._word_embedding_feature_generator = GensimWordEmbeddingsFeatureGenerator(self._db, **params)
        self._word_embedding_feature_generator.execute()
        self._embedding_table_name = self._word_embedding_feature_generator._word_embedding_table_name
        self._words = self._word_embedding_model_creator._word_vector_dict
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GensimWordEmbeddingsFeatureGenerator_mean_posts_content_{}_d0".format(self._embedding_table_name)).attribute_value
        self.assertAlmostEqual(float(db_val), -0.0375, places=4)

    def test_single_word_single_dimension_feature_correctness2(self):
        self._add_post('Post1', 'was')
        self._setup_test()
        # db_val = self._db.get_author_feature(self._author.author_guid,
        #                                      u"word_embeddings_dimension_3_posts_content_max").attribute_value
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_max_posts_content_{}_d3".format(self._embedding_table_name)).attribute_value
        self.assertEqual(db_val, '-0.2133')

    def test_two_words_single_dimension_max_feature_correctness(self):
        self._add_post('Post1', 'was is')
        self._setup_test()
        # db_val = self._db.get_author_feature(self._author.author_guid,
        #                                      u"word_embeddings_dimension_3_posts_content_max").attribute_value
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_max_posts_content_{}_d3".format(self._embedding_table_name)).attribute_value
        was_d3 = self._get_word_dimension('was', 3)
        is_d3 = self._get_word_dimension('is', 3)
        expected_value = max(float(is_d3), float(was_d3))
        self.assertAlmostEqual(float(db_val), (expected_value), places=4)

    def test_two_words_single_dimention_min_feature_correctness(self):
        self._add_post('Post1', 'was is')
        self._setup_test()
        # db_val = self._db.get_author_feature(self._author.author_guid,
        #                                      u"word_embeddings_dimension_201_posts_content_min").attribute_value
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_min_posts_content_{}_d201".format(self._embedding_table_name)).attribute_value
        was_d201 = self._get_word_dimension('was', 201)
        is_d201 = self._get_word_dimension('is', 201)
        expected_value = min(float(is_d201), float(was_d201))
        self.assertAlmostEqual(float(db_val), (expected_value), places=4)

    def test_two_words_single_dimension_np_mean_feature_correctness(self):
        self._add_post('Post1', 'was')
        self._add_post('Post2', 'is')
        self._setup_test()
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_mean_posts_content_{}_d298".format(self._embedding_table_name)).attribute_value
        was_d298 = self._get_word_dimension('was', 298)
        is_d298 = self._get_word_dimension('is', 298)
        expected_value = (float(was_d298) + float(is_d298)) / 2
        self.assertAlmostEqual(float(db_val), expected_value, places=4)

    def test_four_words_single_dimension_np_mean_feature_correctness(self):
        self._add_post('Post1', 'was is')
        self._add_post('Post2', 'with said')
        self._setup_test()
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_mean_posts_content_{}_d150".format(self._embedding_table_name)).attribute_value
        was_d = self._get_word_dimension('was', 150)
        is_d = self._get_word_dimension('is', 150)
        with_d = self._get_word_dimension('with', 150)
        said_d = self._get_word_dimension('said', 150)
        expected_value = (float(was_d) + float(is_d) + float(with_d) + float(said_d)) / 4
        self.assertAlmostEqual(float(db_val), expected_value, places=4)

    def test_single_word_last_dimension_feature_correctness(self):
        self._add_post('Post1', 'was')
        self._setup_test()
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "GloveWordEmbeddingsFeatureGenerator_max_posts_content_{}_d299".format(self._embedding_table_name)).attribute_value
        self.assertAlmostEqual(float(db_val), -0.080348, places=4)

    def _add_post(self, title, content):
        post = Post()
        post.author = self._author.author_full_name
        post.author_guid = self._author.author_guid
        post.content = content
        post.title = title
        post.domain = 'test'
        post.post_id = len(self._posts)
        post.guid = post.post_id
        self._db.addPost(post)
        self._posts.append(post)

    def _set_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'name' + author_guid
        author.name = 'name' + author_guid
        author.domain = 'test'
        author.author_screen_name = 'author_screen_n'
        self._db.add_author(author)
        self._author = author

    def _setup_test(self):
        self._db.session.commit()
        self._word_embedding_model_creator = GloveWordEmbeddingModelCreator(self._db)
        self._word_embedding_model_creator.execute(None)

        params = {'authors': [self._author], 'posts': self._posts}
        self._word_embedding_feature_generator = GloveWordEmbeddingsFeatureGenerator(self._db, **params)
        self._word_embedding_feature_generator.execute()
        self._embedding_table_name = self._word_embedding_feature_generator._word_embedding_table_name
        self._words = self._db.get_word_embedding_dictionary()

    def _get_word_dimension(self, word, dimension):
        word_vec = self._words[word]
        return word_vec[dimension]
