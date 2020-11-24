import unittest
from DB.schema_definition import *
import datetime
from configuration.config_class import getConfig
import random, string
from random import randint
from dataset_builder.clickbait_challenge import clickbait_feature_generator
from dataset_builder.clickbait_challenge.clickbait_feature_generator import Clickbait_Feature_Generator
from collections import Counter

def rand_string_by_len(len):
    is_uppercase = randint(0, 1)
    if is_uppercase == 1:
        return str(''.join(random.sample(string.ascii_lowercase, len)))
    else:
        return str(''.join(random.sample(string.ascii_uppercase, len)))

class click_bait_feature_generator_unittest(unittest.TestCase):
    def setUp(self):
        self._config_parser = getConfig()
        self._domain = str(self._config_parser.get("DEFAULT", "domain"))
        self._db = DB()
        self._db.setUp()

        self._clear_stractures()

        self._counter = 1

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()

    # params for create_article:
    # (name, post_title_length, article_title_length, article_description_length, article_keywords_num,
    #  article_captions_num, article_caption_length, article_paragraph_num, article_paragraph_len,
    #  article_keyboards_length):
    # def test_base_case(self):
    #     author_guid = rand_string_by_len(5)
    #     self._create_article(author_guid, 6, 7, 8, 7, 8, 5, 9, 7, 5)
    #     parameters = {"authors": self._authors, "posts": self._posts_by_authors}
    #     self.cfg = Clickbait_Feature_Generator(self._db, **parameters)
    #     self.cfg.execute()
    #     self._run_tests(unicode('guid_'+author_guid))
    #
    # def test_base_case2(self):
    #     author_guid = rand_string_by_len(5)
    #     self._create_article(author_guid, 1, 3, 2, 1, 1, 1, 1, 1, 1)
    #     parameters = {"authors": self._authors, "posts": self._posts_by_authors}
    #     self.cfg = Clickbait_Feature_Generator(self._db, **parameters)
    #     self.cfg.execute()
    #     self._run_tests(unicode('guid_'+author_guid))
    #
    # def test_base_case3(self):
    #     author_guid = rand_string_by_len(5)
    #     self._create_article(author_guid, 6, 7, 8, 41, 3, 3, 42, 2, 5)
    #     parameters = {"authors": self._authors, "posts": self._posts_by_authors}
    #     self.cfg = Clickbait_Feature_Generator(self._db, **parameters)
    #     self.cfg.execute()
    #     self._run_tests(unicode('guid_'+author_guid))

    # def _create_article_with_texts(self, author_name, post_title, article_title, article_description,
    #                                article_keywords, article_captions, article_paragraphs):
    def test_case_thing(self):
        author_guid = 'test1'
        keywords = 'Tooth, brush, me, look'
        captions = ['THE LAST THING YOU REMMMBER', 'AAA AAA AAAA']
        paragraphs = ['who will last the best from who is that who thing that im the one',
                      'lorem ipsum', 'das fgg, dddd !!!!!']
        self._create_article_by_text(author_guid, 'Log The Things YOU DO WITH THAT', 'test number', 'dddddd',
                                     keywords, captions, paragraphs)
        parameters = {"authors": self._authors, "posts": self._posts_by_authors}
        self.cfg = Clickbait_Feature_Generator(self._db, **parameters)
        self.cfg.execute()
        self._run_tests(str('guid_'+author_guid))

    def _clear_stractures(self):
        self._authors_by_name = {}
        self._posts_by_authors = {}
        self._articles_by_authors = {}
        self._authors = []
        self._posts = []
        self._articles = []
        self._items_by_article = {}

    def _generic_test(self, author_guid, attribute, expected):
        # try:
            attribute = "Clickbait_Feature_Generator_" + attribute
            db_value = self._db.get_author_feature(author_guid, attribute).attribute_value
            self.assertAlmostEqual(float(db_value), float(expected), 6)
        # except Exception as iexec:
        #     print text('failed at author: ' + author_guid + ' attribute: ' + attribute+ ' expected value: ' + unicode(float(expected)) +
        #               ' \nadditional info: '+ iexec.message)
        #     raise Exception(iexec)
        # finally:
            self._db.session.close()

    def _run_tests(self, author_guid):
        self._testNum_of_characters_in_post_title(author_guid)
        self._testNum_of_characters_in_article_title(author_guid)
        self._testNum_of_characters_in_article_description(author_guid)
        self._testAverage_num_of_characters_in_article_keywords(author_guid)
        self._testAverage_num_of_characters_in_article_captions(author_guid)
        self._testAverage_num_of_characters_in_article_paragraphs(author_guid)
        self._test_diff_num_of_characters_post_title_and_article_title(author_guid)
        self._test_diff_num_of_characters_post_title_and_article_description(author_guid)
        self._test_diff_num_of_characters_post_title_and_article_keywords(author_guid)
        self._test_diff_num_of_characters_post_title_and_article_captions(author_guid)
        self._test_diff_num_of_characters_post_title_and_article_paragraphs(author_guid)
        self._test_diff_num_of_characters_article_title_and_article_description(author_guid)
        self._test_diff_num_of_characters_article_title_and_article_keywords(author_guid)
        self._test_diff_num_of_characters_article_title_and_article_captions(author_guid)
        self._test_diff_num_of_characters_article_title_and_article_paragraphs(author_guid)
        self._test_diff_num_of_characters_article_description_and_article_keywords(author_guid)
        self._test_diff_num_of_characters_article_description_and_article_captions(author_guid)
        self._test_diff_num_of_characters_article_description_and_article_paragraphs(author_guid)
        self._test_diff_num_of_characters_article_keywords_and_article_captions(author_guid)
        self._test_diff_num_of_characters_article_keywords_and_article_paragraphs(author_guid)
        self._test_diff_num_of_characters_article_captions_and_article_paragraphs(author_guid)
        self._test_num_of_characters_ratio_article_title_and_post_title(author_guid)
        self._test_num_of_characters_ratio_article_description_and_post_title(author_guid)
        self._test_num_of_characters_ratio_article_keywords_and_post_title(author_guid)
        self._test_num_of_characters_ratio_article_captions_and_post_title(author_guid)
        self._test_num_of_characters_ratio_article_paragraphs_post_title(author_guid)
        self._test_num_of_characters_ratio_article_description_and_article_title(author_guid)
        self._test_num_of_characters_ratio_article_keywords_and_article_title(author_guid)
        self._test_num_of_characters_ratio_article_captions_and_article_title(author_guid)
        self._test_num_of_characters_ratio_article_paragraphs_and_article_title(author_guid)
        self._test_num_of_characters_ratio_article_keywords_and_article_description(author_guid)
        self._test_num_of_characters_ratio_article_captions_and_article_description(author_guid)
        self._test_num_of_characters_ratio_article_paragraphs_and_article_description(author_guid)
        self._test_num_of_characters_ratio_article_captions_and_article_keywords(author_guid)
        self._test_num_of_characters_ratio_article_paragraphs_and_article_keywords(author_guid)
        self._test_num_of_characters_ratio_article_paragraphs_and_article_captions(author_guid)
        self._test_num_of_words_in_article_title(author_guid)
        self._test_num_of_words_in_article_description(author_guid)
        self._test_diff_num_of_words_post_title_and_article_title(author_guid)
        self._test_diff_num_of_words_post_title_and_article_description(author_guid)
        # self._test_diff_num_of_words_post_title_and_article_keywords(author_guid)

# region_tests_implementation
    def _get_post_title(self, author_guid):
        ans = self._posts_by_authors[author_guid][0].content
        return ans

    def _testNum_of_characters_in_post_title(self, author_guid):
        expected_value = self._count_characters_in_sentence(self._get_post_title(author_guid))
        self._generic_test(author_guid, 'num_of_characters_in_post_title', expected_value)

    def _testNum_of_characters_in_article_title(self, author_guid):
        expected_value = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)
        self._generic_test(author_guid, str('num_of_characters_in_article_title'), expected_value)

    def _testNum_of_characters_in_article_description(self, author_guid):
        expected_value = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        self._generic_test(author_guid, str('num_of_characters_in_article_description'), expected_value)

    def _testAverage_num_of_characters_in_article_keywords(self, author_guid):
        expected_value = self._get_avarage_num_of_chars_in_article_keywords(author_guid)
        self._generic_test(author_guid, str('average_num_of_characters_in_article_keywords'), expected_value)

    def _get_avarage_num_of_chars_in_article_keywords(self, author_guid):
        article = self._articles_by_authors[author_guid]
        processed_keywords = article.keywords.replace(" ", "")
        keywords = processed_keywords.split(',')
        if len(keywords) == 0:
            return 0
        lenght = 0
        index = 0
        for word in keywords:
            lenght = lenght + self._count_characters_in_sentence(keywords[index])
            index = index + 1
        return lenght / len(keywords)

    def _testAverage_num_of_characters_in_article_captions(self, author_guid):
        expected_value = self._get_avarage_num_of_chars_in_article_caption(author_guid)
        self._generic_test(author_guid, str('average_num_of_characters_in_article_captions'), expected_value)

    def _get_avarage_num_of_chars_in_article_caption(self, author_guid):
        captions = self._get_items_by_author_and_type(author_guid, 'caption')
        length = 0
        if len(captions) == 0:
            return 0
        for capt in captions:
            length = length + self._count_characters_in_sentence(capt.content)
        ans = float(length) / len(captions)
        return ans

    def _testAverage_num_of_characters_in_article_paragraphs(self, author_guid):
        expected_value = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)
        self._generic_test(author_guid, str('average_num_of_characters_in_article_paragraphs'), expected_value)

    def _get_avarage_num_of_chars_in_article_paragraphs(self, author_guid):
        paragraphs = self._get_items_by_author_and_type(author_guid, 'paragraph')
        length = 0
        if len(paragraphs) == 0:
            return 0
        for paragraph in paragraphs:
            length = length + self._count_characters_in_sentence(paragraph.content)
        return float(length) / len(paragraphs)

    def _get_items_by_author_and_type(self, author_guid, item_type):
        items = self._items_by_article[author_guid]
        item_types = []
        for item in items:
            if item.type == item_type:
                item_types.append(item)
        return item_types

    def _test_diff_num_of_characters_post_title_and_article_title(self, author_guid):
        num_of_chars_in_post = self._count_characters_in_sentence(self._get_post_title(author_guid))

        article = self._articles_by_authors[author_guid]
        num_of_chars_in_article = self._count_characters_in_sentence(article.title)

        expected_value = abs(num_of_chars_in_article - num_of_chars_in_post)
        self._generic_test(author_guid, str('diff_num_of_characters_post_title_and_article_title'), expected_value)

    def _test_diff_num_of_characters_post_title_and_article_description(self, author_guid):
        num_of_chars_in_post_title = self._count_characters_in_sentence(self._get_post_title(author_guid))

        article = self._articles_by_authors[author_guid]
        num_of_chars_in_article_description = self._count_characters_in_sentence(article.description)

        expected_value = abs(num_of_chars_in_article_description - num_of_chars_in_post_title)
        self._generic_test(author_guid, str('diff_num_of_characters_post_title_and_article_description'),
                           expected_value)

    def _test_diff_num_of_characters_post_title_and_article_keywords(self, author_guid):
        num_of_chars_in_post_title = self._count_characters_in_sentence(self._get_post_title(author_guid))

        num_of_chars_in_article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)

        expected_value = abs(num_of_chars_in_post_title - num_of_chars_in_article_keywords)
        self._generic_test(author_guid, str('diff_num_of_characters_post_title_and_article_keywords'),
                           expected_value)

    def _test_diff_num_of_characters_post_title_and_article_captions(self, author_guid):
        num_of_chars_in_post_title = self._count_characters_in_sentence(self._get_post_title(author_guid))

        num_of_chars_in_article_captions = self._get_avarage_num_of_chars_in_article_caption(author_guid)

        expected_value = abs(num_of_chars_in_post_title - num_of_chars_in_article_captions)
        self._generic_test(author_guid, str('diff_num_of_characters_post_title_and_article_captions'),
                           expected_value)

    def _test_diff_num_of_characters_post_title_and_article_paragraphs(self, author_guid):
        num_of_chars_in_post_title = self._count_characters_in_sentence(self._get_post_title(author_guid))
        num_of_chars_in_article_paragraph = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)

        expected_value = abs(num_of_chars_in_post_title - num_of_chars_in_article_paragraph)
        self._generic_test(author_guid, str('diff_num_of_characters_post_title_and_article_paragraphs'),
                           expected_value)

    def _test_diff_num_of_characters_article_title_and_article_description(self, author_guid):
        num_of_chars_in_article_titles = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)
        num_of_chars_in_article_description = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)

        expected_value = abs(num_of_chars_in_article_description - num_of_chars_in_article_titles)
        self._generic_test(author_guid, str('diff_num_of_characters_article_title_and_article_description'),
                           expected_value)

    def _test_diff_num_of_characters_article_title_and_article_keywords(self, author_guid):
        num_of_chars_in_article_title = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)
        avg_article_keyword_length = self._get_avarage_num_of_chars_in_article_keywords(author_guid)

        expected_value = abs(num_of_chars_in_article_title - avg_article_keyword_length)
        self._generic_test(author_guid, str('diff_num_of_characters_article_title_and_article_keywords'),
                           expected_value)

    def _test_diff_num_of_characters_article_title_and_article_captions(self, author_guid):
        num_of_chars_in_article_title = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)
        avg_article_caption = self._get_avarage_num_of_chars_in_article_caption(author_guid)

        expected_value = abs(num_of_chars_in_article_title - avg_article_caption)
        self._generic_test(author_guid, str('diff_num_of_characters_article_title_and_article_captions'),
                           expected_value)

    def _test_diff_num_of_characters_article_title_and_article_paragraphs(self, author_guid):
        num_of_chars_in_article_title = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)
        avg_article_caption = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)

        expected_value = abs(num_of_chars_in_article_title - avg_article_caption)
        self._generic_test(author_guid, str('diff_num_of_characters_article_title_and_article_paragraphs'),
                           expected_value)

    def _test_diff_num_of_characters_article_description_and_article_keywords(self, author_guid):
        num_of_chars_in_article_description = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        avg_article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)

        expected_value = abs(num_of_chars_in_article_description - avg_article_keywords)
        self._generic_test(author_guid, str('diff_num_of_characters_article_description_and_article_keywords'),
                           expected_value)

    def _test_diff_num_of_characters_article_description_and_article_captions(self, author_guid):
        num_of_chars_in_article_description = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        avg_article_captions = self._get_avarage_num_of_chars_in_article_caption(author_guid)

        expected_value = abs(num_of_chars_in_article_description - avg_article_captions)
        self._generic_test(author_guid, str('diff_num_of_characters_article_description_and_article_captions'),
                           expected_value)

    def _test_diff_num_of_characters_article_description_and_article_paragraphs(self, author_guid):
        num_of_chars_in_article_description = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        avg_article_caption = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)

        expected_value = abs(num_of_chars_in_article_description - avg_article_caption)
        self._generic_test(author_guid, str('diff_num_of_characters_article_description_and_article_paragraphs'),
                           expected_value)

    def _test_diff_num_of_characters_article_keywords_and_article_captions(self, author_guid):
        avg_article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)
        avg_article_captions = self._get_avarage_num_of_chars_in_article_caption(author_guid)

        expected_value = abs(avg_article_captions - avg_article_keywords)
        self._generic_test(author_guid, str('diff_num_of_characters_article_keywords_and_article_captions'),
                           expected_value)

    def _test_diff_num_of_characters_article_keywords_and_article_paragraphs(self, author_guid):
        avg_article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)
        avg_article_caption = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)

        expected_value = abs(avg_article_keywords - avg_article_caption)
        self._generic_test(author_guid, str('diff_num_of_characters_article_keywords_and_article_paragraphs'),
                           expected_value)

    def _test_diff_num_of_characters_article_captions_and_article_paragraphs(self, author_guid):
        avg_article_caption = self._get_avarage_num_of_chars_in_article_caption(author_guid)
        avg_article_paragraph = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)

        expected_value = abs(avg_article_caption-avg_article_paragraph)
        self._generic_test(author_guid, str('diff_num_of_characters_article_captions_and_article_paragraphs'),
                           expected_value)

    def _test_num_of_characters_ratio_article_title_and_post_title(self, author_guid):
        post_title_len = self._count_characters_in_sentence(self._get_post_title(author_guid))
        article_title_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)

        expected_value = float(article_title_len)/float(post_title_len)
        self._generic_test(author_guid, str('num_of_characters_ratio_article_title_and_post_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_description_and_post_title(self, author_guid):
        article_description_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        post_title_len = self._count_characters_in_sentence(self._get_post_title(author_guid))

        expected_value = article_description_len/float(post_title_len)
        self._generic_test(author_guid, str('num_of_characters_ratio_article_description_and_post_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_keywords_and_post_title(self, author_guid):
        article_keywords_len = self._get_avarage_num_of_chars_in_article_keywords(author_guid)
        post_title_len = self._count_characters_in_sentence(self._get_post_title(author_guid))

        expected_value = article_keywords_len/float(post_title_len)
        self._generic_test(author_guid, str('num_of_characters_ratio_article_keywords_and_post_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_captions_and_post_title(self, author_guid):
        article_caption_len = self._get_avarage_num_of_chars_in_article_caption(author_guid)
        post_title_len = self._count_characters_in_sentence(self._get_post_title(author_guid))

        expected_value = float(article_caption_len)/post_title_len
        self._generic_test(author_guid, str('num_of_characters_ratio_article_captions_and_post_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_paragraphs_post_title(self, author_guid):
        article_paragraph_len = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)
        post_title_len = self._count_characters_in_sentence(self._get_post_title(author_guid))

        expected_value = float(article_paragraph_len)/post_title_len
        self._generic_test(author_guid, str('num_of_characters_ratio_article_paragraphs_post_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_description_and_article_title(self, author_guid):
        article_description_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        article_title_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)

        expected_value = float(article_description_len)/article_title_len
        self._generic_test(author_guid, str('num_of_characters_ratio_article_description_and_article_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_keywords_and_article_title(self, author_guid):
        article_keyword_length = self._get_avarage_num_of_chars_in_article_keywords(author_guid)
        article_title_length = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)

        expected_value = float(article_keyword_length)/article_title_length
        self._generic_test(author_guid, str('num_of_characters_ratio_article_keywords_and_article_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_captions_and_article_title(self, author_guid):
        article_caption_length = self._get_avarage_num_of_chars_in_article_caption(author_guid)
        artcle_title_length = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)

        expected_value = float(article_caption_length)/artcle_title_length
        self._generic_test(author_guid, str('num_of_characters_ratio_article_captions_and_article_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_paragraphs_and_article_title(self, author_guid):
        article_paragraph_len = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)
        article_title_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].title)

        expected_value = float(article_paragraph_len)/article_title_len
        self._generic_test(author_guid, str('num_of_characters_ratio_article_paragraphs_and_article_title'),
                           expected_value)

    def _test_num_of_characters_ratio_article_keywords_and_article_description(self, author_guid):
        article_keywords_len = self._get_avarage_num_of_chars_in_article_keywords(author_guid)
        article_descrtiption_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)

        expected_value = float(article_keywords_len)/article_descrtiption_len
        self._generic_test(author_guid, str('num_of_characters_ratio_article_keywords_and_article_description'),
                           expected_value)

    def _test_num_of_characters_ratio_article_captions_and_article_description(self, author_guid):
        article_caption_len = self._get_avarage_num_of_chars_in_article_caption(author_guid)
        article_description_len = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)

        expected_value = float(article_caption_len)/article_description_len
        self._generic_test(author_guid, str('num_of_characters_ratio_article_captions_and_article_description'),
                           expected_value)

    def _test_num_of_characters_ratio_article_paragraphs_and_article_description(self, author_guid):
        article_description = self._count_characters_in_sentence(self._articles_by_authors[author_guid].description)
        article_paragraph_len = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)

        expected_value = float(article_paragraph_len)/article_description
        self._generic_test(author_guid, str('num_of_characters_ratio_article_paragraphs_and_article_description'),
                           expected_value)

    def _test_num_of_characters_ratio_article_captions_and_article_keywords(self, author_guid):
        article_captions = self._get_avarage_num_of_chars_in_article_caption(author_guid)
        article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)

        expected_value = float(article_captions)/article_keywords
        self._generic_test(author_guid, str('num_of_characters_ratio_article_captions_and_article_keywords'),
                           expected_value)

    def _test_num_of_characters_ratio_article_paragraphs_and_article_keywords(self, author_guid):
        article_paragraphs = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)
        article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)

        expected_value = float(article_paragraphs)/article_keywords
        self._generic_test(author_guid, str('num_of_characters_ratio_article_paragraphs_and_article_keywords'),
                           expected_value)

    def _test_num_of_characters_ratio_article_paragraphs_and_article_captions(self, author_guid):
        article_paragraphs = self._get_avarage_num_of_chars_in_article_paragraphs(author_guid)
        article_captions = self._get_avarage_num_of_chars_in_article_caption(author_guid)

        expected_value = float(article_paragraphs)/article_captions
        self._generic_test(author_guid, str('num_of_characters_ratio_article_paragraphs_and_article_captions'),
                           expected_value)

    def _test_num_of_words_in_post_title(self, author_guid):
        post_title = self._get_post_title(author_guid)
        expected_value = self._get_num_of_distinct_words(post_title)
        self._generic_test(author_guid, str('num_of_words_in_post_title'),expected_value)

    def _test_num_of_words_in_article_title(self, author_guid):
        article_title = self._articles_by_authors[author_guid].title
        expected_value = self._get_num_of_distinct_words(article_title)
        self._generic_test(author_guid, str('num_of_words_in_article_title'), expected_value)

    def _test_num_of_words_in_article_description(self, author_guid):
        article_description = self._articles_by_authors[author_guid].description
        expected_value = self._get_num_of_distinct_words(article_description)
        self._generic_test(author_guid, str('num_of_words_in_article_description'), expected_value)

    def _test_diff_num_of_words_post_title_and_article_title(self, author_guid):
        post_title = self._get_post_title(author_guid)
        words_in_post_title = self._get_num_of_distinct_words(post_title)

        article_title = self._articles_by_authors[author_guid].title
        words_in_article_title = self._get_num_of_distinct_words(article_title)

        expected_value = abs(words_in_post_title-words_in_article_title)
        self._generic_test(author_guid, str('diff_num_of_words_post_title_and_article_title'), expected_value)

    def _test_diff_num_of_words_post_title_and_article_description(self, author_guid):
        post_title = self._get_post_title(author_guid)
        words_in_post_title = self._get_num_of_distinct_words(post_title)

        article_description = self._articles_by_authors[author_guid].description
        words_in_article_description = self._get_num_of_distinct_words(article_description)

        expected_value = abs(words_in_post_title-words_in_article_description)
        self._generic_test(author_guid, str('diff_num_of_words_post_title_and_article_description'), expected_value)

    def _test_diff_num_of_words_post_title_and_article_keywords(self, author_guid):
        post_title = self._get_post_title(author_guid)
        words_in_post_title = self._get_num_of_distinct_words(post_title)

        article_keywords = self._get_avarage_num_of_chars_in_article_keywords(author_guid)

        expected_value = abs(words_in_post_title-article_keywords)
        self._generic_test(author_guid, str('diff_num_of_words_post_title_and_article_keywords'), expected_value)
    # endregion

    def _get_num_of_distinct_words(self, text):
        distinct_words = Counter(text.split())
        return len(distinct_words)

# region_create_data_stracture
    def _create_article(self, name, post_title_length, article_title_length,
                        article_description_length, article_keywords_num,
                        article_captions_num, article_caption_length, article_paragraph_num, article_paragraph_len,
                        article_keyboards_length):
        self._create_author(name)
        self._create_post(name, post_title_length)
        self._create_article_specific(name, article_title_length, article_description_length, article_keywords_num,
                                      article_keyboards_length)
        self._create_items(name, str('caption'), article_captions_num, article_caption_length)
        self._create_items(name, str('paragraph'), article_paragraph_num, article_paragraph_len)

    def _create_article_with_texts(self, author_name, post_title, article_title, article_description,
                                   article_keywords, article_captions, article_paragraphs):
        self._create_author(author_name)
        self._create_post_with_text(author_name, post_title)
        self._create_article_by_text(author_name,article_title, article_description, article_keywords,
                                               article_captions, article_paragraphs)


    def _create_author(self, name):
        author = Author()
        author.name = str(name)
        author.domain = 'Article'
        author.author_guid = str('guid_' + name)
        author.author_screen_name = str(name)
        author.author_full_name = str('full_name_' + name)
        author.statuses_count = 10
        author.author_osn_id = 1
        author.created_at = datetime.datetime.strptime('2016-04-02 15:43:00', '%Y-%m-%d %H:%M:%S')
        author.missing_data_complementor_insertion_date = datetime.datetime.now()
        author.xml_importer_insertion_date = datetime.datetime.now()
        author.author_type = 1
        self._authors_by_name[name] = author
        self._authors.append(author)
        self._db.add_author(author)

    def _create_post_with_text(self, author_name, post_title):
        post = Post()
        post.post_id = str(author_name)
        post.author = str(author_name)
        post.title = ''
        post.guid = self._authors_by_name[author_name].author_guid
        post.url = str('post_url' + author_name)
        tempDate = '2016-05-05 00:00:00'
        day = datetime.timedelta(1)
        post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day * self._counter
        post.domain = 'Article'
        post.author_guid = self._authors_by_name[author_name].author_guid
        post.content = post_title
        post.xml_importer_insertion_date = datetime.datetime.now()
        self._posts.append(post)
        self._db.addPost(post)
        self._posts = []
        self._posts.append(post)
        self._posts_by_authors[self._authors_by_name[author_name].author_guid] = self._posts

    def _create_post(self, name, post_title_length):
        post = Post()
        post.post_id = str(name)
        post.author = str(name)
        post.title = ''
        post.guid = self._authors_by_name[name].author_guid
        post.url = str('post_url' + name)
        tempDate = '2016-05-05 00:00:00'
        day = datetime.timedelta(1)
        post.date = datetime.datetime.strptime(tempDate, '%Y-%m-%d %H:%M:%S') + day * self._counter
        post.domain = 'Article'
        post.author_guid = self._authors_by_name[name].author_guid
        post.content = str(rand_string_by_len(post_title_length))
        post.xml_importer_insertion_date = datetime.datetime.now()
        self._posts.append(post)
        self._db.addPost(post)
        self._posts = []
        self._posts.append(post)
        self._posts_by_authors[self._authors_by_name[name].author_guid] = self._posts

    def _create_article_specific(self, name, article_title_length, article_description_length, article_keywords_num,
                                 article_keyboards_length):
        article = Target_Article()
        article.post_id = name
        # article.title = rand_string_by_len(article_title_length)
        article.title = 'are you \"THE\" thing?'
        # article.description = rand_string_by_len(article_description_length)
        article.description = 'the quick brown fox jumps over the lazy dog'
        article.keywords = self._create_keywords(article_keywords_num, article_keyboards_length)
        self._db.add_target_articles([article])
        self._articles.append(article)
        self._articles_by_authors[self._authors_by_name[name].author_guid] = article
        self._items_by_article[self._authors_by_name[name].author_guid] = []

    def _create_article_by_text(self, author_name, post_title, article_title, article_description, article_keywords,
                                article_captions, article_paragraphs):
        article = Target_Article()
        article.post_id = author_name
        article.author_guid = author_name
        # article.title = rand_string_by_len(article_title_length)
        article.title = article_title
        # article.description = rand_string_by_len(article_description_length)
        article.description = article_description
        article.keywords = article_keywords
        self._create_author(author_name)
        self._create_post_with_text(author_name, post_title)
        self._db.add_target_articles([article])
        self._articles.append(article)
        self._articles_by_authors[self._authors_by_name[author_name].author_guid] = article
        self._items_by_article[self._authors_by_name[author_name].author_guid] = []
        self._create_items_by_text(author_name, 'caption', article_captions)
        self._create_items_by_text(author_name, 'paragraph', article_paragraphs)

    def _create_keywords(self, article_keywords_num, article_keyboards_length):
        keywords = ''
        for i in range(article_keywords_num):
            keywords = keywords + rand_string_by_len(article_keyboards_length) + ','
        return keywords[:len(keywords) - 1]

    def _create_items_by_text(self, author_name, item_type, items_text):
        counter = 0
        for text in items_text:
            artItem = Target_Article_Item()
            artItem.post_id = author_name
            artItem.author_guid = author_name
            artItem.item_number = counter
            artItem.type = item_type
            artItem.content = text
            self._items_by_article[self._authors_by_name[author_name].author_guid].append(artItem)
            counter = counter + 1
        self._db.addPosts(self._items_by_article[self._authors_by_name[author_name].author_guid])

    def _create_items(self, name, item_type, item_num, item_len):
        for i in range(item_num):
            artItem = Target_Article_Item()
            artItem.post_id = name
            artItem.item_number = 0
            artItem.type = item_type
            artItem.content = rand_string_by_len(item_len)
            self._items_by_article[self._authors_by_name[name].author_guid].append(artItem)
            self._db.addPost(artItem)
        pass
    pass
    #endregion
    def _count_characters_in_sentence(self, sentence):
        clean_sentence = sentence.replace(" ", "")
        return len(clean_sentence)