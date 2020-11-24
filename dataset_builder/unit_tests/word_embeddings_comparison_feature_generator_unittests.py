import unittest
from DB.schema_definition import *
from dataset_builder.feature_extractor.word_embeddings_comparison_feature_generator import \
    Word_Embeddings_Comparison_Feature_Generator
from dataset_builder.word_embedding.glove_word_embedding_model_creator import GloveWordEmbeddingModelCreator
import commons

class Word_Embeddings_Comparison_Feature_Generator_Unittests(unittest.TestCase):
    def setUp(self):
        self._config_parser = getConfig()
        self._db = DB()
        self._db.setUp()

        #deleteme
        # self._db = self._db.fill_author_type_by_post_type()
        #deleteme
        self._is_load_wikipedia_300d_glove_model = True
        self._wikipedia_model_file_path = "data/input/glove/test_glove.6B.300d.txt"
        self._table_name = "wikipedia_model_300d"
        self._word_vector_dict_full_path = "data/output/word_embedding/"
        self._word_vector_dict = {}

        self._author = None
        self._set_author('test_user')
        self._posts = []

    def tearDown(self):
        self._db.session.close()

    def test_simple_case(self):
        self._add_post('of to a', 'of the')
        #
        self._setup_test()
        of_vec_d1 = self._get_word_dimension('of',1)
        the_vec_d1 = self._get_word_dimension('the',1)
        max_val = max(of_vec_d1, the_vec_d1)
        min_val = min(of_vec_d1, the_vec_d1)
        db_val = self._db.get_author_feature(self._author.author_guid,
                                              "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_content_min_TO_posts_content_max_d1").attribute_value
        expected_val = min_val - max_val
        self.assertAlmostEqual(float(db_val), float(expected_val), places=4)

        of_vec_d150 = self._get_word_dimension('of',150)
        the_vec_d150 = self._get_word_dimension('the',150)
        max_val = max(of_vec_d150, the_vec_d150)
        min_val = min(of_vec_d150, the_vec_d150)
        db_val = self._db.get_author_feature(self._author.author_guid,
                                              "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_content_min_TO_posts_content_max_d150").attribute_value
        expected_val = min_val - max_val
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

    def test_no_words(self):
        self._add_post('Title1', '')
        self._setup_test()
        db_val = self._db.get_author_feature(self._author.author_guid, "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_content_min_TO_posts_content_max_d1").attribute_value
        self.assertEqual(float(db_val), 0.0) # need to create real assert

    def test_two_posts(self):
        self._add_post('Title1', ' the')
        self._add_post('Title2', 'of')
        self._add_post('Title3', '')
        self._setup_test()
        of_vec_d1 = self._get_word_dimension('of', 1)
        the_vec_d1 = self._get_word_dimension('the', 1)
        max_val = max(of_vec_d1, the_vec_d1)
        min_val = min(of_vec_d1, the_vec_d1)
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_content_min_TO_posts_content_max_d1").attribute_value
        expected_val = min_val - max_val
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

        of_vec_d150 = self._get_word_dimension('of', 150)
        the_vec_d150 = self._get_word_dimension('the', 150)
        max_val = max(of_vec_d150, the_vec_d150)
        min_val = min(of_vec_d150, the_vec_d150)
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_content_min_TO_posts_content_max_d150").attribute_value
        expected_val = min_val - max_val
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

    def test_title_to_content(self):
        self._add_post('the', 'the')
        self._add_post('of', 'of')
        self._add_post('', '')
        self._setup_test()
        of_vec_d1 = self._get_word_dimension('of', 1)
        the_vec_d1 = self._get_word_dimension('the', 1)
        max_val = max(of_vec_d1, the_vec_d1)
        min_val = min(of_vec_d1, the_vec_d1)
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_title_min_TO_posts_content_max_d1").attribute_value
        expected_val = min_val - max_val
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

    def test_distance_functions(self):
        self._add_post('of', 'the')
        self._setup_test()
        of_vec = self._words['of']
        the_vec = self._words['the']

        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_title_min_TO_posts_content_max_DISTANCE-FUNCTION_jaccard_index").attribute_value
        expected_val = jaccard_index(of_vec, the_vec)
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_title_min_TO_posts_content_max_DISTANCE-FUNCTION_euclidean_distance").attribute_value
        expected_val = euclidean_distance(of_vec, the_vec)
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_title_min_TO_posts_content_max_DISTANCE-FUNCTION_euclidean_distance").attribute_value
        expected_val = euclidean_distance(of_vec, the_vec)
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_title_min_TO_posts_content_max_DISTANCE-FUNCTION_manhattan_distance").attribute_value
        expected_val = manhattan_distance(of_vec, the_vec)
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embeddings_Comparison_Feature_Generator_subtraction_posts_title_min_TO_posts_content_max_DISTANCE-FUNCTION_minkowski_distance").attribute_value
        expected_val = minkowski_distance(of_vec, the_vec)
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

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
        self._db.add_author(author)
        self._author = author

    def _setup_test(self):
        self._db.session.commit()
        self._word_embedding_model_creator = GloveWordEmbeddingModelCreator(self._db)
        self._word_embedding_model_creator.execute(None)

        params = {'authors': [self._author], 'posts': self._posts}
        self._word_embedding_feature_generator = Word_Embeddings_Comparison_Feature_Generator(self._db, **params)
        self._word_embedding_feature_generator.execute()

        self._words = self._db.get_word_embedding_dictionary()

    def _get_word_dimension(self, word, dimension):
        word_vec = self._words[word]
        return word_vec[dimension]
