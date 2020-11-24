#Written by Lior Bass 3/27/2018
import unittest

from DB.schema_definition import DB, Post, Author, Target_Article, Target_Article_Item
from commons import commons
from configuration.config_class import getConfig
from dataset_builder.word_embedding.word_embedding_differential_feature_generator import Word_Embedding_Differential_Feature_Generator
from dataset_builder.word_embedding.glove_word_embedding_model_creator import GloveWordEmbeddingModelCreator


class Word_Embedding_Differential_Feature_Generator_Unittests(unittest.TestCase):
    def setUp(self):
        self._config_parser = getConfig()
        self._db = DB()
        self._db.setUp()

        self._model = Word_Embedding_Differential_Feature_Generator(self._db)

        self._posts=[]
        self._author = None
        self._set_author('test_user')

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()


    def test_simple_case(self):
        self._add_post('of to a for', 'of is')
        self._add_target_article('0','of was ','am that was')
        self._setup_test()

        is_vec1 = self._get_word_dimension('is', 0)
        was_vec_d1 = self._get_word_dimension('was', 0)
        expected_val = was_vec_d1-is_vec1
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embedding_Differential_Feature_Generator_differential_sum_target_articles_title_to_posts_content_d0").attribute_value
        self.assertAlmostEqual(float(db_val), expected_val, places=4)
        expected_val = was_vec_d1 - is_vec1
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embedding_Differential_Feature_Generator_differential_np.mean_target_articles_title_to_posts_content_d0").attribute_value
        self.assertAlmostEqual(float(db_val), expected_val, places=4)
        is_vec = self._words['is']
        was_vec = self._words['was']
        expected_val = commons.euclidean_distance(is_vec,was_vec)
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embedding_Differential_Feature_Generator_differential_distance_function_euclidean_distance_target_articles_title_np.mean_TO_posts_content_np.mean").attribute_value
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

    def test_few_words(self):
        self._add_post('of to a for', 'of is on')
        self._add_target_article('0','of was that','am that was')
        self._setup_test()

        dimension = 0
        is_vec1 = self._get_word_dimension('is', dimension)
        on_vec1 = self._get_word_dimension('on',dimension)
        tot1 = is_vec1+on_vec1
        was_vec_d1 = self._get_word_dimension('was', dimension)
        that_vec_d1 = self._get_word_dimension('that', dimension)
        tot2 = was_vec_d1+that_vec_d1
        expected_val = tot2-tot1
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embedding_Differential_Feature_Generator_differential_sum_target_articles_title_to_posts_content_d"+str(dimension)).attribute_value
        self.assertAlmostEqual(float(db_val), expected_val, places=4)
        dimension = 140
        is_vec1 = self._get_word_dimension('is', dimension)
        on_vec1 = self._get_word_dimension('on',dimension)
        tot1 = is_vec1+on_vec1
        was_vec_d1 = self._get_word_dimension('was', dimension)
        that_vec_d1 = self._get_word_dimension('that', dimension)
        tot2 = was_vec_d1+that_vec_d1
        expected_val = tot2-tot1
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embedding_Differential_Feature_Generator_differential_sum_target_articles_title_to_posts_content_d"+str(dimension)).attribute_value
        self.assertAlmostEqual(float(db_val), expected_val, places=4)

    def test_opposite(self):
        self._add_post('am that was','of was that')
        self._add_target_article('0', 'of is on','of to a for')
        self._setup_test()

        dimension = 0
        is_vec1 = self._get_word_dimension('is', dimension)
        on_vec1 = self._get_word_dimension('on',dimension)
        tot1 = is_vec1+on_vec1
        was_vec_d1 = self._get_word_dimension('was', dimension)
        that_vec_d1 = self._get_word_dimension('that', dimension)
        tot2 = was_vec_d1+that_vec_d1
        expected_val = tot1-tot2
        db_val = self._db.get_author_feature(self._author.author_guid,
                                             "Word_Embedding_Differential_Feature_Generator_differential_sum_target_articles_title_to_posts_content_d"+str(dimension)).attribute_value
        self.assertAlmostEqual(float(db_val), expected_val, places=4)
    def test_empty_word(self):
        self._add_post('of to a for', '')
        self._add_target_article('0','of was that','am that was')
        self._setup_test()
        self.assertTrue(True)

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
        self._model = Word_Embedding_Differential_Feature_Generator(self._db, **params)
        self._model.execute()

        self._words = self._db.get_word_embedding_dictionary()


    def _get_word_dimension(self, word, dimension):
        word_vec = self._words[word]
        return word_vec[dimension]

    def _add_target_article(self, post_id, title, description):
        target_article = Target_Article()
        target_article.author_guid = 'test_user'
        target_article.post_id = post_id
        target_article.title = title
        target_article.description = description
        target_article.keywords = 'temp, lala, fafa'
        self._db.add_target_articles([target_article])

    def _add_target_article_item(self, post_id, type, content, author_guid):
        article_item = Target_Article_Item()
        article_item.post_id = post_id
        article_item.type = type
        article_item.item_number = 3
        article_item.content = content
        article_item.author_guid = author_guid
        self._db.addPosts([article_item])