from unittest import TestCase
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from DB.schema_definition import DB, Author, Post, Claim_Tweet_Connection, Target_Article_Item, Target_Article
from configuration.config_class import getConfig
from dataset_builder.word_embedding.glove_word_embedding_model_creator import GloveWordEmbeddingModelCreator
from dataset_builder.word_embedding.gensim_word_embedding_trainer import GensimWordEmbeddingsModelTrainer


class TestGensimWordEmbeddingsModelTrainer(TestCase):
    def setUp(self):
        self._config_parser = getConfig()
        self._db = DB()
        self._db.setUp()

        # self._Word_Embedding_Model_Creator.execute(None)
        self._is_load_wikipedia_300d_glove_model = True
        self._wikipedia_model_file_path = "data/input/glove/test_glove.6B.300d_small.txt"
        self._table_name = "wikipedia_model_300d"
        self._word_vector_dict_full_path = "data/output/word_embedding/"
        self._word_vector_dict = {}

        self._author = None
        self._set_author('test_user')
        self._counter = 0
        self._posts = []

    def tearDown(self):
        self._db.session.close()

    def test_add_additional_fields_to_existing_table(self):
        self._add_post('was', 'is')
        self._add_post('is', 'was')
        self._db.session.commit()
        self._word_embedding_model_creator = GensimWordEmbeddingsModelTrainer(self._db)

        self._word_embedding_model_creator.execute(None)
        self._word_embedding_model_creator._aggregation_functions_names = ['sum']
        self._word_embedding_model_creator.execute(None)

        file_output_path = self._word_embedding_model_creator._saved_models_path + self._word_embedding_model_creator._table_name + ".csv"
        data = pd.DataFrame.from_csv(file_output_path)

        word_embedding_results = data.loc[(data['author_id'] == 'test_user') & (data['table_name'] == 'posts') & (
                data['targeted_field_name'] == 'content')]
        sum_value_df = word_embedding_results.loc[word_embedding_results['word_embedding_type'] == 'sum']
        mean_value_df = word_embedding_results.loc[word_embedding_results['word_embedding_type'] == 'np.mean']

        try:
            if len(sum_value_df.values.tolist()) > 0 and len(mean_value_df.values.tolist()) > 0:
                self.assertTrue(True)
            else:
                self.fail()
        except:
            self.fail()

    def test_case_post_represent_by_posts(self):
        self._add_post('post1', 'the claim', 'Claim')
        self._add_post('post2', 'dog cat pig man')  # 2
        self._add_post('post3', 'TV is the best guys')  # 1
        self._add_claim_tweet_connection('post1', 'post2')
        self._add_claim_tweet_connection('post1', 'post3')
        self._db.session.commit()
        self._word_embedding_model_creator = GensimWordEmbeddingsModelTrainer(self._db)
        self._word_embedding_model_creator._targeted_fields_for_embedding = [{
            'source': {'table_name': 'posts', 'id': 'post_id'},
            'connection': {'table_name': 'claim_tweet_connection', 'source_id': 'claim_id', 'target_id': 'post_id'},
            'destination': {'table_name': 'posts', 'id': 'post_id', 'target_field': 'content',
                            "where_clauses": []}}]

        self._word_embedding_model_creator.execute(None)
        model_name_path = self._word_embedding_model_creator._prepare_model_name_path()
        model = Word2Vec.load(model_name_path)
        word_vector_dict = self._word_embedding_model_creator._get_word_embedding_dict(model)
        self._words = word_vector_dict
        self._words_vectors = self._get_posts_val()
        expected_val = self._calc_results()
        self._generic_test(expected_val, 'post1')

    def _setup_test(self):
        self._db.session.commit()
        self._word_embedding_model_creator = GensimWordEmbeddingsModelTrainer(self._db)
        self._word_embedding_model_creator.execute(None)

        self._words = self._db.get_word_embedding_dictionary()
        self._words_vectors = self._get_posts_val()

    def _generic_test(self, expected_value, source_id=""):
        if source_id == "":
            source_id = self._author.author_guid

        file_output_path = self._word_embedding_model_creator._saved_models_path + self._word_embedding_model_creator._table_name + ".csv"
        data = pd.DataFrame.from_csv(file_output_path)

        word_embedding_results = data.loc[(data['author_id'] == source_id) & (data['table_name'] == 'posts') & (data['targeted_field_name'] == 'content')]

        self.assert_word_embedding(word_embedding_results, expected_value, 'min')
        self.assert_word_embedding(word_embedding_results, expected_value, 'max')
        self.assert_word_embedding(word_embedding_results, expected_value, 'np.mean')

    def assert_word_embedding(self, db_results, expected_value, type):
        result_value = db_results.loc[db_results['word_embedding_type'] == type, '0':].values.tolist()[0]
        self.assertEqual(list(expected_value[type]), result_value)

    def _generic_non_equal_test(self, expected_value):
        db_results = self._db.get_author_word_embedding(self._author.author_guid, 'posts', 'content')
        self.assertNotEqual(expected_value['min'], db_results['min'])
        self.assertNotEqual(expected_value['max'], db_results['max'])
        self.assertNotEqual(expected_value['np.mean'], db_results['np.mean'])

    def _set_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'name' + author_guid
        author.name = 'name' + author_guid
        author.domain = 'test'
        self._db.add_author(author)
        self._author = author

    def _add_post(self, title, content, _domain='Microblog'):
        post = Post()
        post.author = self._author.author_full_name
        post.author_guid = self._author.author_guid
        post.content = content
        post.title = title
        post.domain = _domain
        post.post_id = title
        post.guid = title
        self._db.addPost(post)
        self._posts.append(post)

    def _get_posts_val(self):  # return the vectors for all the words in the added posts
        vals = {}
        for post in self._posts:
            for word in post.content.split():
                if word in list(self._words.keys()):
                    vals[word] = self._words[word]
        return list(vals.values())

    def _calc_mean(self, vectors):
        vectors = self._get_posts_val()
        if len(vectors) == 0:
            return (0,) * 300
        ziped_vec = list(zip(*vectors))
        result = list(map(eval('np.mean'), ziped_vec))
        return tuple(result)

    def _calc_min(self, vectors):
        vectors = self._get_posts_val()
        if len(vectors) == 0:
            return (0,) * 300
        ziped_vec = list(zip(*vectors))
        result = list(map(eval('min'), ziped_vec))
        return tuple(result)

    def _calc_max(self, vectors):
        vectors = self._get_posts_val()
        if len(vectors) == 0:
            return (0,) * 300
        ziped_vec = list(zip(*vectors))
        result = list(map(eval('max'), ziped_vec))
        return tuple(result)

    def _calc_results(self):
        vectors = self._words_vectors
        results = {}
        results['min'] = self._calc_min(vectors)
        results['max'] = self._calc_max(vectors)
        results['np.mean'] = self._calc_mean(vectors)
        return results

    def _add_target_article(self, post_id, title, description, author_guid):
        target_article = Target_Article()
        target_article.author_guid = author_guid
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

    def _add_claim_tweet_connection(self, claim_id, post_id):
        connection = Claim_Tweet_Connection()
        connection.claim_id = claim_id
        connection.post_id = post_id
        self._db.add_claim_connections([connection])
        pass
