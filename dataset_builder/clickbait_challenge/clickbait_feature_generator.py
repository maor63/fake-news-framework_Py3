
from commons.commons import *
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
import nltk
from nltk import word_tokenize
from nltk.tag import map_tag
from nltk.corpus import words
from nltk.corpus import stopwords

import re

class Clickbait_Feature_Generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)

        # author_screen_name = post_id
        # author = post in clickbait
        self._author_screen_name_targeted_article_dict = self._create_post_id_targeted_article_dict()
        self._author_screen_name_targeted_article_item_dict = self._create_post_id_targeted_article_item_dict()
        self._author_screen_name_text_image_dict = self._create_post_id_text_image()
        self._stopwords_file = 'data/english_stopwords.txt'
        self._load_stop_words()
        self._words = words.words()

    def cleanUp(self):
        pass

    def _string_len_without_spaces(self, str):
        return len(''.join(str.split(' ')))

    def _get_post_by_author(self, kwargs):
        author = kwargs['author']
        author_guid = author.author_guid
        posts = self.author_guid_posts_dict[author_guid]
        post = posts[0]
        return post

    def num_of_characters_in_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            post_title = self._get_post_title(kwargs)
            length_title = self._string_len_without_spaces(post_title)
            return length_title

    def num_of_characters_in_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            author_screen_name = author.author_screen_name
            targeted_article = self._author_screen_name_targeted_article_dict[author_screen_name]
            targeted_article_title = targeted_article.title
            num_of_characters_targeted_article_title = self._string_len_without_spaces(targeted_article_title)
            return num_of_characters_targeted_article_title

    def num_of_characters_in_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            targeted_article_description = self._get_article_description(kwargs)
            num_of_characters_in_article_description = self._string_len_without_spaces(targeted_article_description)
            return num_of_characters_in_article_description

    def average_num_of_characters_in_article_keywords(self, **kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        article_keywords = list(article_keywords_set)
        average_num_of_characters = self._calculate_average_article_characters(article_keywords)
        return average_num_of_characters

    def average_num_of_characters_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            average_num_of_characters = self._calculate_average_article_characters(article_captions)
            return average_num_of_characters
        return -1

    def average_num_of_characters_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            average_num_of_characters = self._calculate_average_article_characters(article_paragraphs)
            return average_num_of_characters
        return -1
    '''
    -1 -> no image
    0 - inf -> the length of the characters extracted by OCR
    '''
    def num_of_characters_in_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            post_image_text = self._get_post_image_text(kwargs)
            if post_image_text is not None:
                text_image_content = post_image_text.content
                num_of_characters_text_image = len(text_image_content)
                return num_of_characters_text_image
            else:
                return -1

    def _calculate_average_article_characters(self, article_items):
        sum_of_characters = 0
        for article_item in article_items:
            num_of_characters_article_item = self._string_len_without_spaces(article_item)
            sum_of_characters += num_of_characters_article_item
        average_num_of_characters = sum_of_characters / float(len(article_items))
        return average_num_of_characters


    '''
    ABS(#characters(post title) - #characters(article title))
    '''
    def diff_num_of_characters_post_title_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_targeted_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            diff_num_of_characters = abs(num_of_characters_targeted_article_title - num_of_characters_post_title)
            return diff_num_of_characters

    '''
    ABS(#characters(post title) - #characters(article description))
    '''

    def diff_num_of_characters_post_title_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_targeted_article_description = self.num_of_characters_in_article_description(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            diff_num_of_characters = abs(num_of_characters_targeted_article_description - num_of_characters_post_title)
            return diff_num_of_characters

    '''
    ABS(#characters(post title) - #characters(article keywords))
    '''

    def diff_num_of_characters_post_title_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            average_num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            diff_num_of_characters = abs(average_num_of_characters_article_keywords - num_of_characters_post_title)
            return diff_num_of_characters

    '''
    ABS(#characters(post title) - #characters(article captions))
    '''

    def diff_num_of_characters_post_title_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            average_num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            diff_num_of_characters = abs(average_num_of_characters_article_captions - num_of_characters_post_title)
            return diff_num_of_characters

    '''
    ABS(#characters(post title) - #characters(article paragraphs))
    '''

    def diff_num_of_characters_post_title_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            average_num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            diff_num_of_characters = abs(average_num_of_characters_article_paragraphs - num_of_characters_post_title)
            return diff_num_of_characters

    '''
    ABS(#characters(post title) - #characters(post_image_text))
    '''

    def diff_num_of_characters_post_title_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            average_num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            diff_num_of_characters = abs(average_num_of_characters_post_image_text - num_of_characters_post_title)
            return diff_num_of_characters



    def diff_num_of_characters_article_title_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_title - num_of_characters_article_description)
            return diff_num_of_characters

    def diff_num_of_characters_article_title_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_title - num_of_characters_article_keywords)
            return diff_num_of_characters

    def diff_num_of_characters_article_title_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_title - num_of_characters_article_captions)
            return diff_num_of_characters

    def diff_num_of_characters_article_title_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_title - num_of_characters_article_paragraphs)
            return diff_num_of_characters

    def diff_num_of_characters_article_title_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_title - num_of_characters_post_image_text)
            return diff_num_of_characters

    #-------------------------------------------------------------------------------


    def diff_num_of_characters_article_description_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_description - num_of_characters_article_keywords)
            return diff_num_of_characters

    def diff_num_of_characters_article_description_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_description - num_of_characters_article_captions)
            return diff_num_of_characters

    def diff_num_of_characters_article_description_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_description - num_of_characters_article_paragraphs)
            return diff_num_of_characters

    def diff_num_of_characters_article_description_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_description - num_of_characters_post_image_text)
            return diff_num_of_characters

    #---------------------------------------------------------------------------------

    def diff_num_of_characters_article_keywords_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_keywords - num_of_characters_article_captions)
            return diff_num_of_characters

    def diff_num_of_characters_article_keywords_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_keywords - num_of_characters_article_paragraphs)
            return diff_num_of_characters

    def diff_num_of_characters_article_keywords_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_keywords - num_of_characters_post_image_text)
            return diff_num_of_characters


    #---------------------------------------------------------------------------------


    def diff_num_of_characters_article_captions_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_captions - num_of_characters_article_paragraphs)
            return diff_num_of_characters

    def diff_num_of_characters_article_captions_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_captions - num_of_characters_post_image_text)
            return diff_num_of_characters

    def diff_num_of_characters_article_paragraphs_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)

            diff_num_of_characters = abs(num_of_characters_article_paragraphs - num_of_characters_post_image_text)
            return diff_num_of_characters

    #----------------------------------------------------------------------------------

    def num_of_characters_ratio_article_title_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_targeted_article_title = self.num_of_characters_in_article_title(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            if num_of_characters_targeted_article_title > 0 and num_of_characters_post_title > 0:
                num_of_characters_ratio = abs(num_of_characters_targeted_article_title / float(num_of_characters_post_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_description_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)
            num_of_characters_targeted_article_description = self.num_of_characters_in_article_description(**kwargs)

            if num_of_characters_targeted_article_description > 0 and num_of_characters_post_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_targeted_article_description / float(num_of_characters_post_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_keywords_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            if num_of_characters_article_keywords > 0 and num_of_characters_post_title > 0:
                num_of_characters_ratio = abs(num_of_characters_article_keywords / float(num_of_characters_post_title))
                return num_of_characters_ratio
            return -1


    def num_of_characters_ratio_article_captions_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_captions= self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            if num_of_characters_article_captions > 0 and num_of_characters_post_title > 0:
                num_of_characters_ratio = abs(num_of_characters_article_captions / float(num_of_characters_post_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_paragraphs_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            if num_of_characters_article_paragraphs > 0 and num_of_characters_post_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_article_paragraphs / float(num_of_characters_post_title))
                return num_of_characters_ratio
            return -1


    def num_of_characters_ratio_post_image_text_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_post_title = self.num_of_characters_in_post_title(**kwargs)

            if num_of_characters_post_image_text > 0 and num_of_characters_post_title > 0:
                num_of_characters_ratio = abs(num_of_characters_post_image_text / float(num_of_characters_post_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_description_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)

            if num_of_characters_article_description > 0 and num_of_characters_article_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_article_description / float(num_of_characters_article_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_keywords_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)

            if num_of_characters_article_keywords > 0 and num_of_characters_article_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_article_keywords / float(num_of_characters_article_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_captions_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)

            if num_of_characters_article_captions > 0 and num_of_characters_article_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_article_captions / float(num_of_characters_article_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_paragraphs_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)

            if num_of_characters_article_paragraphs > 0 and num_of_characters_article_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_article_paragraphs / float(num_of_characters_article_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_post_image_text_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_article_title = self.num_of_characters_in_article_title(**kwargs)

            if num_of_characters_post_image_text > 0 and num_of_characters_article_title > 0:
                num_of_characters_ratio = abs(
                    num_of_characters_post_image_text / float(num_of_characters_article_title))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_keywords_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)

            if num_of_characters_article_description > 0 and num_of_characters_article_keywords > 0:
                num_of_characters_ratio = abs(num_of_characters_article_keywords / float(num_of_characters_article_description))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_captions_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)

            if num_of_characters_article_description > 0 and num_of_characters_article_captions > 0:
                num_of_characters_ratio = abs(num_of_characters_article_captions / float(num_of_characters_article_description))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_paragraphs_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)

            if num_of_characters_article_description > 0 and num_of_characters_article_paragraphs > 0:
                num_of_characters_ratio = abs(num_of_characters_article_paragraphs / float(num_of_characters_article_description))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_post_image_text_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_article_description = self.num_of_characters_in_article_description(**kwargs)

            if num_of_characters_article_description > 0 and num_of_characters_article_paragraphs > 0:
                num_of_characters_ratio = abs(num_of_characters_article_paragraphs / float(num_of_characters_article_description))
                return num_of_characters_ratio
            return -1


    #-----------------------------------------------------------------------

    def num_of_characters_ratio_article_captions_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)

            if num_of_characters_article_keywords > 0 and num_of_characters_article_captions > 0:
                num_of_characters_ratio = abs(num_of_characters_article_captions / float(num_of_characters_article_keywords))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_paragraphs_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)

            if num_of_characters_article_keywords > 0 and num_of_characters_article_paragraphs > 0:
                num_of_characters_ratio = abs(num_of_characters_article_paragraphs / float(num_of_characters_article_keywords))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_post_image_text_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_article_keywords = self.average_num_of_characters_in_article_keywords(**kwargs)

            if num_of_characters_article_keywords > 0 and num_of_characters_post_image_text > 0:
                num_of_characters_ratio = abs(num_of_characters_post_image_text / float(num_of_characters_article_keywords))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_article_paragraphs_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)

            if num_of_characters_article_captions > 0 and num_of_characters_article_paragraphs > 0:
                num_of_characters_ratio = abs(num_of_characters_article_paragraphs / float(num_of_characters_article_captions))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_post_image_text_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_article_captions = self.average_num_of_characters_in_article_captions(**kwargs)

            if num_of_characters_article_captions > 0 and num_of_characters_post_image_text > 0:
                num_of_characters_ratio = abs(num_of_characters_post_image_text / float(
                    num_of_characters_article_captions))
                return num_of_characters_ratio
            return -1

    def num_of_characters_ratio_post_image_text_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_characters_post_image_text = self.num_of_characters_in_post_image_text(**kwargs)
            num_of_characters_article_paragraphs = self.average_num_of_characters_in_article_paragraphs(**kwargs)

            if num_of_characters_article_paragraphs > 0 and num_of_characters_post_image_text > 0:
                num_of_characters_ratio = abs(num_of_characters_post_image_text / float(
                    num_of_characters_article_paragraphs))
                return num_of_characters_ratio
            return -1

    #---------------------------------------------------------------------------------
    # WORDS
    # --------------------------------------------------------------------------------
    def num_of_words_in_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            post_title = self._get_post_title(kwargs)

            words = get_words_by_content(post_title)
            num_of_words = len(words)
            return num_of_words

    def num_of_words_in_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            author_screen_name = author.author_screen_name
            targeted_article = self._author_screen_name_targeted_article_dict[author_screen_name]
            targeted_article_title = targeted_article.title

            targeted_article_title_words = get_words_by_content(targeted_article_title)
            num_of_words_targeted_article_title = len(targeted_article_title_words)
            return num_of_words_targeted_article_title

    def num_of_words_in_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            author_screen_name = author.author_screen_name
            targeted_article = self._author_screen_name_targeted_article_dict[author_screen_name]
            targeted_article_description = targeted_article.description
            words_in_article_description = get_words_by_content(targeted_article_description)
            num_of_words_in_article_description = len(words_in_article_description)
            return num_of_words_in_article_description

    def _calculate_average_article_words(self, article_items):
        sum_of_words = 0
        for article_item in article_items:
            article_item_words = get_words_by_content(article_item)
            article_item_words = list(article_item_words)
            num_of_words_in_article_item = len(article_item_words)
            sum_of_words += num_of_words_in_article_item
        average = sum_of_words / float(len(article_items))
        return average

    def average_num_of_words_in_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            article_keywords_set = self._get_article_keywords_set(kwargs)
            article_keywords = list(article_keywords_set)
            average = self._calculate_average_article_words(article_keywords)
            return average

    def average_num_of_words_in_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            targeted_article_captions = self._get_article_captions(kwargs)
            if targeted_article_captions is not None:
                average = self._calculate_average_article_words(targeted_article_captions)
                return average
            return -1

    def average_num_of_words_in_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            targeted_article_paragraphs = self._get_article_paragraphs(kwargs)
            if targeted_article_paragraphs is not None:
                average = self._calculate_average_article_words(targeted_article_paragraphs)
                return average
            return -1
    '''
        -1 -> no image
        0 - inf -> the length of the words extracted by OCR
        '''

    def num_of_words_in_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            post_image_text = self._get_post_image_text(kwargs)
            if post_image_text is not None:
                text_image = post_image_text.content
                words_in_text_image = get_words_by_content(text_image)
                num_of_words_text_image = len(words_in_text_image)
                return num_of_words_text_image
            else:
                return -1

    # def _get_words_by_content(self, content):
    #     words = []
    #     tokenizer = SpaceTokenizer()
    #     words += tokenizer.tokenize(content)
    #     #words = list(set(words))
    #     words = frozenset(words)
    #     return words

    '''
    ABS(#words(post title) - #words(article title))
    '''

    def diff_num_of_words_post_title_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)

            diff = abs(num_of_words_article_title - num_of_words_post_title)
            return diff

    '''
    ABS(#words(post title) - #words(article title))
    '''
    def diff_num_of_words_post_title_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)

            diff = abs(num_of_words_article_description - num_of_words_post_title)
            return diff

    '''
    ABS(#words(post title) - #words(article keywords))
    '''
    def diff_num_of_words_post_title_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)
            num_of_words_article_keywords= self.average_num_of_words_in_article_keywords(**kwargs)

            diff = abs(num_of_words_article_keywords - num_of_words_post_title)
            return diff

    def diff_num_of_words_post_title_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)

            diff = abs(num_of_words_article_captions - num_of_words_post_title)
            return diff

    def diff_num_of_words_post_title_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)

            diff = abs(num_of_words_article_paragraphs - num_of_words_post_title)
            return diff

    def diff_num_of_words_post_title_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)

            diff = abs(num_of_words_post_image_text - num_of_words_post_title)
            return diff

    def diff_num_of_words_article_title_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_article_title(**kwargs)
            num_of_words_description = self.num_of_words_in_article_description(**kwargs)

            diff = abs(num_of_words_description - num_of_words_post_title)
            return diff

    def diff_num_of_words_article_title_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_article_title(**kwargs)
            num_of_words_keywords = self.average_num_of_words_in_article_keywords(**kwargs)

            diff = abs(num_of_words_keywords - num_of_words_post_title)
            return diff

    def diff_num_of_words_article_title_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_article_title(**kwargs)
            num_of_words_captions = self.average_num_of_words_in_article_captions(**kwargs)

            diff = abs(num_of_words_captions - num_of_words_post_title)
            return diff

    def diff_num_of_words_article_title_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_article_title(**kwargs)
            num_of_words_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)

            diff = abs(num_of_words_paragraphs - num_of_words_post_title)
            return diff

    def diff_num_of_words_article_title_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_title = self.num_of_words_in_article_title(**kwargs)
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)

            diff = abs(num_of_words_post_image_text - num_of_words_post_title)
            return diff

    def diff_num_of_words_article_description_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_description = self.num_of_words_in_article_description(**kwargs)
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)

            diff = abs(num_of_words_article_keywords - num_of_words_post_description)
            return diff

    def diff_num_of_words_article_description_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_description = self.num_of_words_in_article_description(**kwargs)
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)

            diff = abs(num_of_words_article_captions - num_of_words_post_description)
            return diff

    def diff_num_of_words_article_description_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_description = self.num_of_words_in_article_description(**kwargs)
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)

            diff = abs(num_of_words_article_paragraphs - num_of_words_post_description)
            return diff

    def diff_num_of_words_article_description_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_description = self.num_of_words_in_article_description(**kwargs)
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)

            diff = abs(num_of_words_post_image_text - num_of_words_post_description)
            return diff

    def diff_num_of_words_article_keywords_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)

            diff = abs(num_of_words_article_captions - num_of_words_article_keywords)
            return diff

    def diff_num_of_words_article_keywords_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)

            diff = abs(num_of_words_article_paragraphs - num_of_words_article_keywords)
            return diff

    def diff_num_of_words_article_keywords_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)

            diff = abs(num_of_words_post_image_text - num_of_words_article_keywords)
            return diff

    def diff_num_of_words_article_captions_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)

            diff = abs(num_of_words_article_paragraphs - num_of_words_article_captions)
            return diff

    def diff_num_of_words_article_captions_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)

            diff = abs(num_of_words_post_image_text - num_of_words_article_captions)
            return diff

    def diff_num_of_words_article_paragraphs_and_post_image_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)

            diff = abs(num_of_words_post_image_text - num_of_words_article_paragraphs)
            return diff

    def num_of_words_ratio_article_title_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)

            if num_of_words_article_title > 0 and num_of_words_post_title > 0:
                num_of_words_ratio = abs(num_of_words_article_title / float(num_of_words_post_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_description_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)

            if num_of_words_article_description > 0 and num_of_words_post_title > 0:
                num_of_words_ratio = abs(num_of_words_article_description / float(num_of_words_post_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_keywords_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)

            if num_of_words_article_keywords > 0 and num_of_words_post_title > 0:
                num_of_words_ratio = abs(num_of_words_article_keywords / float(num_of_words_post_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_captions_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)

            if num_of_words_article_captions > 0 and num_of_words_post_title > 0:
                num_of_words_ratio = abs(num_of_words_article_captions / float(num_of_words_post_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_paragraphs_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)

            if num_of_words_article_paragraphs > 0 and num_of_words_post_title > 0:
                num_of_words_ratio = abs(num_of_words_article_paragraphs / float(num_of_words_post_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_post_image_text_and_post_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)
            num_of_words_post_title = self.num_of_words_in_post_title(**kwargs)

            if num_of_words_post_image_text > 0 and num_of_words_post_title > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_post_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_description_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)

            if num_of_words_article_description > 0 and num_of_words_article_title > 0:
                num_of_words_ratio = abs(num_of_words_article_description / float(num_of_words_article_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_keywords_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)

            if num_of_words_article_keywords > 0 and num_of_words_article_title > 0:
                num_of_words_ratio = abs(num_of_words_article_keywords / float(num_of_words_article_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_captions_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)

            if num_of_words_article_captions > 0 and num_of_words_article_title > 0:
                num_of_words_ratio = abs(num_of_words_article_captions / float(num_of_words_article_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_paragraphs_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)

            if num_of_words_article_paragraphs > 0 and num_of_words_article_title > 0:
                num_of_words_ratio = abs(num_of_words_article_paragraphs / float(num_of_words_article_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_post_image_text_and_article_title(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)
            num_of_words_article_title = self.num_of_words_in_article_title(**kwargs)

            if num_of_words_post_image_text > 0 and num_of_words_article_title > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_article_title))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_keywords_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.average_num_of_words_in_article_keywords(**kwargs)
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)

            if num_of_words_article_description > 0 and num_of_words_post_image_text > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_article_description))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_captions_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)

            if num_of_words_article_description > 0 and num_of_words_article_captions > 0:
                num_of_words_ratio = abs(num_of_words_article_captions / float(num_of_words_article_description))
                return num_of_words_ratio
            return -1
        
    def num_of_words_ratio_article_paragraphs_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)

            if num_of_words_article_description > 0 and num_of_words_article_paragraphs > 0:
                num_of_words_ratio = abs(num_of_words_article_paragraphs / float(num_of_words_article_description))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_post_image_text_and_article_description(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)
            num_of_words_article_description = self.num_of_words_in_article_description(**kwargs)

            if num_of_words_article_description > 0 and num_of_words_post_image_text > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_article_description))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_captions_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)

            if num_of_words_article_keywords > 0 and num_of_words_article_captions > 0:
                num_of_words_ratio = abs(num_of_words_article_captions / float(num_of_words_article_keywords))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_paragraphs_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)

            if num_of_words_article_keywords > 0 and num_of_words_article_captions > 0:
                num_of_words_ratio = abs(num_of_words_article_captions / float(num_of_words_article_keywords))
                return num_of_words_ratio
            return -1
    def num_of_words_ratio_post_image_text_and_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)
            num_of_words_article_keywords = self.average_num_of_words_in_article_keywords(**kwargs)

            if num_of_words_article_keywords > 0 and num_of_words_post_image_text > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_article_keywords))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_article_paragraphs_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)

            if num_of_words_article_captions > 0 and num_of_words_article_paragraphs > 0:
                num_of_words_ratio = num_of_words_article_paragraphs / float(num_of_words_article_captions)
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_post_image_text_and_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)
            num_of_words_article_captions = self.average_num_of_words_in_article_captions(**kwargs)

            if num_of_words_article_captions > 0 and num_of_words_post_image_text > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_article_captions))
                return num_of_words_ratio
            return -1

    def num_of_words_ratio_post_image_text_and_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            num_of_words_post_image_text = self.num_of_words_in_post_image_text(**kwargs)
            num_of_words_article_paragraphs = self.average_num_of_words_in_article_paragraphs(**kwargs)

            if num_of_words_article_paragraphs > 0 and num_of_words_post_image_text > 0:
                num_of_words_ratio = abs(num_of_words_post_image_text / float(num_of_words_article_paragraphs))
                return num_of_words_ratio
            return -1
    #---------------------------------------------------------------------------------

    def _get_targeted_article(self, kwargs):
        author = kwargs['author']
        author_screen_name = author.author_screen_name
        targeted_article = self._author_screen_name_targeted_article_dict[author_screen_name]
        return targeted_article

    def _get_article_keywords_set(self, kwargs):
        targeted_article = self._get_targeted_article(kwargs)
        article_keywords_str = targeted_article.keywords.replace(" ","")
        article_keywords = article_keywords_str.split(",")
        article_keywords_set = set(article_keywords)
        return article_keywords_set

    def _get_article_keywords_string(self, kwargs):
        targeted_article = self._get_targeted_article(kwargs)
        article_keywords_str = targeted_article.keywords
        return article_keywords_str

    def num_of_article_keywords_exist_in_post_title(self, **kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        post_title = self._get_post_title(kwargs)
        post_title_words = get_words_by_content(post_title)

        num_of_common_words = self._find_num_of_common_words(post_title_words, article_keywords_set)
        return num_of_common_words

    def num_of_article_keywords_exist_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)

        article_keywords_set = self._get_article_keywords_set(kwargs)

        article_title_words = get_words_by_content(article_title)

        num_of_common_words = self._find_num_of_common_words(article_title_words, article_keywords_set)
        return num_of_common_words

    def num_of_article_keywords_exist_in_article_description(self, **kwargs):
        targeted_article = self._get_targeted_article(kwargs)
        article_description = targeted_article.description

        article_keywords_set = self._get_article_keywords_set(kwargs)

        article_description_words = get_words_by_content(article_description)

        num_of_common_words = self._find_num_of_common_words(article_description_words, article_keywords_set)
        return num_of_common_words

    def num_of_article_keywords_exist_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            num_of_common_words = self._calculate_common_words_between_article_keywords_and_article_items(article_captions, kwargs)
            return num_of_common_words
        return -1

    def num_of_article_keywords_exist_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            num_of_common_words = self._calculate_common_words_between_article_keywords_and_article_items(article_paragraphs, kwargs)
            return num_of_common_words
        return -1

    def _find_num_of_common_words(self, targeted_words, word_set):
        num_of_common_words = 0
        for targeted_word in targeted_words:
            if targeted_word in word_set:
                num_of_common_words += 1
        return num_of_common_words

    def post_creation_hour(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            post = self._get_post_by_author(kwargs)
            creation_date = post.date
            creation_date_str = date_to_str(creation_date)
            date_hour = creation_date_str.split(" ")
            hour = date_hour[1]
            hour_parts = hour.split(":")
            creation_hour = hour_parts[0]
            creation_hour = int(creation_hour)
            return creation_hour

    def _create_post_id_targeted_article_dict(self):
        targetd_articles = self._db.get_targeted_articles()

        post_id_targeted_article_dict = {}
        for targetd_article in targetd_articles:
            post_id = targetd_article.post_id
            post_id_targeted_article_dict[post_id] = targetd_article
        return post_id_targeted_article_dict

    def _create_post_id_text_image(self):
        text_images = self._db.get_text_images()

        post_id_text_image_dict = {}
        for text_image in text_images:
            post_id = text_image.post_id
            post_id_text_image_dict[post_id] = text_image
        return post_id_text_image_dict

    def _create_post_id_targeted_article_item_dict(self):
        targeted_article_items = self._db.get_targeted_article_items()
        post_id_targeted_article_item_dict = {}
        for targeted_article_item in targeted_article_items:
            post_id = targeted_article_item.post_id
            item_type = targeted_article_item.type
            item_number = targeted_article_item.item_number
            item_content = targeted_article_item.content

            if post_id not in post_id_targeted_article_item_dict:
                post_id_targeted_article_item_dict[post_id] = {}
                post_id_targeted_article_item_dict[post_id][item_type] = []
                post_id_targeted_article_item_dict[post_id][item_type].append(item_content)
            elif item_type not in post_id_targeted_article_item_dict[post_id]:
                post_id_targeted_article_item_dict[post_id][item_type] = []
                post_id_targeted_article_item_dict[post_id][item_type].append(item_content)
            else:
                post_id_targeted_article_item_dict[post_id][item_type].append(item_content)
        return post_id_targeted_article_item_dict

    def num_of_article_paragraphs(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            article_paragraphs = self._get_article_paragraphs(kwargs)
            if article_paragraphs is not None:
                num_of_article_paragraphs = len(article_paragraphs)
                return num_of_article_paragraphs
            return -1

    def _get_article_paragraphs(self, kwargs):
        author = kwargs['author']
        author_screen_name = author.author_screen_name
        targeted_item_type = 'paragraph'
        if author_screen_name in self._author_screen_name_targeted_article_item_dict and \
                targeted_item_type in self._author_screen_name_targeted_article_item_dict[author_screen_name]:
            targeted_article_paragraphs = self._author_screen_name_targeted_article_item_dict[author_screen_name][targeted_item_type]
            return targeted_article_paragraphs
        return None

    def _get_post_image_text(self, kwargs):
        author = kwargs['author']
        author_screen_name = author.author_screen_name
        if author_screen_name in self._author_screen_name_text_image_dict:
            text_image = self._author_screen_name_text_image_dict[author_screen_name]
            return text_image
        return None


    def num_of_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            article_captions = self._get_article_captions(kwargs)
            if article_captions is not None:
                num_article_captions = len(article_captions)
                return num_article_captions
            return -1

    def num_of_characters_in_article_captions(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            targeted_article_captions = self._get_article_captions(kwargs)
            if targeted_article_captions is not None:
                sum_of_characters_captions = 0
                for caption in targeted_article_captions:
                    num_of_characters_caption = len(caption)
                    sum_of_characters_captions += num_of_characters_caption
                average = sum_of_characters_captions / float(len(targeted_article_captions))
                return average
            return -1

    def _get_article_captions(self, kwargs):
        author = kwargs['author']
        author_screen_name = author.author_screen_name
        targeted_item_type = 'caption'
        if author_screen_name in self._author_screen_name_targeted_article_item_dict and \
            targeted_item_type in self._author_screen_name_targeted_article_item_dict[author_screen_name]:
            targeted_article_captions = self._author_screen_name_targeted_article_item_dict[author_screen_name][targeted_item_type]
            return targeted_article_captions
        else:
            return None

    def num_of_article_keywords(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            keywords_set = self._get_article_keywords_set(kwargs)
            keywords = list(keywords_set)
            num_of_article_keywords = len(keywords)
            return num_of_article_keywords

    # def average_num_of_article_keywords_exist_in_article_captions(self, **kwargs):
    #     article_keywords_set = self._get_article_keywords_set(kwargs)
    #     article_captions = self._get_article_captions(kwargs)
    #     if article_captions is not None:
    #         sum_of_common_words = 0
    #         for article_caption in article_captions:
    #             article_caption_words = self._get_words_by_content(article_caption)
    #             num_of_common_words = self._find_num_of_common_words(article_caption_words, article_keywords_set)
    #             sum_of_common_words += num_of_common_words
    #         average_common_words = sum_of_common_words / float(len(article_captions))
    #         return average_common_words
    #     return -1

    def num_of_at_signs_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = '@'
        num_of_at_signs = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_at_signs

    def num_of_at_signs_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = '@'
        num_of_at_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_at_signs

    def num_of_at_signs_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        if article_description is not None:
            selected_sign = '@'
            num_of_at_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_description)
            return num_of_at_signs
        return -1

    def num_of_at_signs_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            selected_sign = '@'
            num_of_at_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_captions)
            return num_of_at_signs
        return -1

    def num_of_at_signs_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            selected_sign = '@'
            num_of_at_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_paragraphs)
            return num_of_at_signs
        return -1


    def _get_post_title(self, kwargs):
        post = self._get_post_by_author(kwargs)
        post_title = post.content
        return post_title

    def num_of_number_signs_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = '#'
        num_of_number_signs = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_number_signs

    def num_of_number_signs_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = '#'
        num_of_number_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_number_signs

    def num_of_number_signs_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        if article_description is not None:
            selected_sign = '#'
            num_of_number_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_description)
            return num_of_number_signs
        return -1

    def num_of_number_signs_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            selected_sign = '#'
            num_of_number_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_captions)
            return num_of_number_signs
        return -1

    def num_of_number_signs_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            selected_sign = '#'
            num_of_number_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_paragraphs)
            return num_of_number_signs
        return -1

    def _calculate_num_of_given_signs_by_content(self, selected_sign, content):
        num_of_signs = content.count(selected_sign)
        return num_of_signs

    def _get_article_title(self, kwargs):
        article = self._get_targeted_article(kwargs)
        article_title = article.title
        return article_title

    def num_of_question_marks_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = '?'
        num_of_question_marks = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_question_marks

    def num_of_question_marks_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = '?'
        num_of_question_marks = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_question_marks

    def num_of_question_marks_in_article_description(self, **kwargs):
        article_title = self._get_article_description(kwargs)
        selected_sign = '?'
        num_of_question_marks = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_question_marks


    def num_of_question_marks_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        selected_sign = '?'
        if article_captions is not None:
            sum_of_question_marks = self._calculate_num_of_given_signs_in_article_items(selected_sign, article_captions)
            return sum_of_question_marks
        return -1

    def num_of_question_marks_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        selected_sign = '?'
        if article_paragraphs is not None:
            sum_of_question_marks = self._calculate_num_of_given_signs_in_article_items(selected_sign, article_paragraphs)
            return sum_of_question_marks
        return -1

    def _get_article_description(self, kwargs):
        article = self._get_targeted_article(kwargs)
        article_description = article.description
        return article_description

    def num_of_colon_signs_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = ':'
        num_of_colons = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_colons

    def num_of_colon_signs_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = ':'
        num_of_colons = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_colons

    def num_of_colon_signs_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        if article_description is not None:
            selected_sign = ':'
            num_of_colons = self._calculate_num_of_given_signs_by_content(selected_sign, article_description)
            return num_of_colons
        return -1

    def num_of_colon_signs_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            selected_sign = ':'
            num_of_colon_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_captions)
            return num_of_colon_signs
        return -1

    def num_of_colon_signs_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            selected_sign = ':'
            num_of_colon_signs= self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_paragraphs)
            return num_of_colon_signs
        return -1

    def num_of_retweet_signs_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = 'RT'
        num_of_retweets_signs = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_retweets_signs

    def num_of_retweet_signs_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = 'RT'
        num_of_retweets_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_retweets_signs

    def num_of_retweet_signs_in_article_description(self, **kwargs):
        article_title = self._get_article_description(kwargs)
        if article_title is not None:
            selected_sign = 'RT'
            num_of_retweets_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
            return num_of_retweets_signs
        return -1

    def num_of_retweet_signs_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            selected_sign = 'RT'
            num_of_retweet_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_captions)
            return num_of_retweet_signs
        return -1

    def num_of_retweet_signs_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            selected_sign = 'RT'
            num_of_retweet_signs = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_paragraphs)
            return num_of_retweet_signs
        return -1

    def num_of_commas_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = ','
        num_of_commas = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_commas

    def num_of_commas_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = ','
        num_of_commas = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_commas

    def num_of_commas_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        if article_description is not None:
            selected_sign = ','
            num_of_commas = self._calculate_num_of_given_signs_by_content(selected_sign, article_description)
            return num_of_commas
        return -1

    def num_of_commas_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            selected_sign = ','
            num_of_commas = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_captions)
            return num_of_commas
        return -1

    def num_of_commas_in_article_pargaraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            selected_sign = ','
            num_of_commas = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_paragraphs)
            return num_of_commas
        return -1


    def _calculate_num_of_selected_sign_in_article_items(self, selected_sign, article_items):
        if article_items is not None:
            sum_of_selected_signs = self._calculate_num_of_given_signs_in_article_items(selected_sign, article_items)
            return sum_of_selected_signs
        return -1

    def _calculate_num_of_given_signs_in_article_items(self, selected_sign, article_items):
        sum = 0
        for article_item in article_items:
            num_of_selected_signs = self._calculate_num_of_given_signs_by_content(selected_sign, article_item)
            sum += num_of_selected_signs
        return sum

    def _calculate_common_words_between_article_keywords_and_article_items(self, article_items, kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        total_article_item_words = []
        for article_item in article_items:
            article_itemn_words = get_words_by_content(article_item)
            article_itemn_words = list(article_itemn_words)
            total_article_item_words.extend(article_itemn_words)

        total_article_item_words = list(set(total_article_item_words))
        num_of_common_words = self._find_num_of_common_words(total_article_item_words, article_keywords_set)
        return num_of_common_words

    def num_of_ellipsis_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        selected_sign = '...'
        num_of_ellipsis = self._calculate_num_of_given_signs_by_content(selected_sign, post_title)
        return num_of_ellipsis

    def num_of_ellipsis_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        selected_sign = '...'
        num_of_ellipsis = self._calculate_num_of_given_signs_by_content(selected_sign, article_title)
        return num_of_ellipsis

    def num_of_ellipsis_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        if article_description is not None:
            selected_sign = '...'
            num_of_ellipsis = self._calculate_num_of_given_signs_by_content(selected_sign, article_description)
            return num_of_ellipsis
        return -1

    def num_of_ellipsis_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        if article_captions is not None:
            selected_sign = '...'
            num_of_ellipsis = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_captions)
            return num_of_ellipsis
        return -1

    def num_of_ellipsis_in_article_pargaraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        if article_paragraphs is not None:
            selected_sign = '...'
            num_of_ellipsis = self._calculate_num_of_selected_sign_in_article_items(selected_sign, article_paragraphs)
            return num_of_ellipsis
        return -1




    ######################################################################################
    # ENGLISH DICTIONARY FEATURES
    ######################################################################################

    def num_of_formal_words_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        num_of_formal_words = self._get_num_of_formal_words_by_item(post_title)
        return num_of_formal_words

    def num_of_formal_words_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        num_of_formal_words = self._get_num_of_formal_words_by_item(article_title)
        return num_of_formal_words

    def num_of_formal_words_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        num_of_formal_words = self._get_num_of_formal_words_by_item(article_description)
        return num_of_formal_words

    def num_of_formal_words_in_article_keywords(self, **kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        article_keywords = list(article_keywords_set)
        num_of_formal_words = self._get_num_of_formal_words_from_language_dictionary(article_keywords)
        return num_of_formal_words

    def num_of_formal_words_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        num_of_formal_words = self._calculate_num_of_formal_words_in_aricle_items(article_captions)
        return num_of_formal_words

    def num_of_formal_words_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        num_of_formal_words = self._calculate_num_of_formal_words_in_aricle_items(article_paragraphs)
        return num_of_formal_words

    def num_of_formal_words_in_post_image_text(self, **kwargs):
        post_image_text = self._get_post_image_text(kwargs)
        if post_image_text is not None:
            post_image_text_content = post_image_text.content
            num_of_formal_words = self._get_num_of_formal_words_by_item(post_image_text_content)
            return num_of_formal_words
        return -1

    def num_of_informal_words_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        num_of_formal_words = self._get_num_of_informal_words_by_item(post_title)
        return num_of_formal_words

    def num_of_informal_words_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        num_of_informal_words = self._get_num_of_informal_words_by_item(article_title)
        return num_of_informal_words

    def num_of_informal_words_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        num_of_informal_words = self._get_num_of_informal_words_by_item(article_description)
        return num_of_informal_words


    def num_of_informal_words_in_article_keywords(self, **kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        #article_keywords = list(article_keywords_set)
        num_of_informal_words = self._get_num_of_informal_words_from_language_dictionary(article_keywords_set)
        return num_of_informal_words

    def num_of_informal_words_in_article_captions(self, **kwargs):
        article_captions = self._get_article_captions(kwargs)
        num_of_informal_words = self._calculate_num_of_informal_words_in_aricle_items(article_captions)
        return num_of_informal_words

    def num_of_informal_words_in_article_paragraphs(self, **kwargs):
        article_paragraphs = self._get_article_paragraphs(kwargs)
        num_of_informal_words = self._calculate_num_of_informal_words_in_aricle_items(article_paragraphs)
        return num_of_informal_words

    def num_of_informal_words_in_post_image_text(self, **kwargs):
        post_image_text = self._get_post_image_text(kwargs)
        if post_image_text is not None:
            post_image_text_content = post_image_text.content
            num_of_informal_words = self._get_num_of_informal_words_by_item(post_image_text_content)
            return num_of_informal_words
        return -1

    def _calculate_num_of_formal_words_in_aricle_items(self, article_items):
        if article_items is not None:
            sum = 0
            num_of_article_items = len(article_items)
            for article_item in article_items:
                article_item_words = get_words_by_content(article_item)
                num_of_formal_words = self._get_num_of_informal_words_from_language_dictionary(article_item_words)
                sum += num_of_formal_words
            average = sum / num_of_article_items
            return average
        return -1

    def _calculate_num_of_informal_words_in_aricle_items(self, article_items):
        if article_items is not None:
            sum = 0
            num_of_article_items = len(article_items)
            for article_item in article_items:
                article_item_words = get_words_by_content(article_item)
                num_of_formal_words = self._get_num_of_informal_words_from_language_dictionary(article_item_words)
                sum += num_of_formal_words
            average = sum / num_of_article_items
            return average
        return -1


    def _get_num_of_formal_words_by_item(self, item):
        words = self._get_words_by_given_item(item)
        num_of_formal_words = self._get_num_of_formal_words_from_language_dictionary(words)
        return num_of_formal_words

    # item could be post_title, article_title, article_description, etc.
    def _get_words_by_given_item(self, item):
        word_set = get_words_by_content(item)
        words = list(word_set)
        return words

    def _get_post_title_words(self, kwargs):
        post_title = self._get_post_title(kwargs)
        word_set = get_words_by_content(post_title)
        words = list(word_set)
        return words

    def percent_of_formal_words_in_post_title(self, **kwargs):
        words = self._get_post_title_words(kwargs)
        percent = self._calculate_percent_of_formal_words(words)
        return percent

    def percent_of_informal_words_in_post_title(self, **kwargs):
        words = self._get_post_title_words(kwargs)
        percent = self._calculate_percent_of_informal_words(words)
        return percent

    # def _get_num_of_formal_words_from_language_dictionary(self, words):
    #     clean_words = self._clean_words(words)
    #     words_with_meaning = self._check_meaning_words(clean_words)
    #     counter = 0
    #     for word in words_with_meaning:
    #         if words_with_meaning[word] != {'temp': None}:
    #             counter = counter + 1
    #     return counter

    def _get_num_of_formal_words_from_language_dictionary(self, word_set):
        clean_words = self._clean_words(word_set)
        counter = 0
        for word in clean_words:
            if word in self._words:
                counter = counter + 1
        return counter

    def _clean_words(self, words):
        clean_words = []
        for word in words:
            clean_words.append(self._clean_word(word))
        return clean_words

    def percent_of_formal_words_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        word_set = get_words_by_content(article_title)
        percent = self._calculate_percent_of_formal_words(word_set)
        return percent

    def percent_of_informal_words_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        word_set = get_words_by_content(article_title)
        percent = self._calculate_percent_of_informal_words(word_set)
        return percent

    def percent_of_formal_words_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        word_set = get_words_by_content(article_description)
        percent = self._calculate_percent_of_formal_words(word_set)
        return percent

    def percent_of_informal_words_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        word_set = get_words_by_content(article_description)
        percent = self._calculate_percent_of_informal_words(word_set)
        return percent


    def percent_of_formal_words_in_article_keywords(self, **kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        percent = self._calculate_percent_of_formal_words(article_keywords_set)
        return percent

    def percent_of_informal_words_in_article_keywords(self, **kwargs):
        article_keywords_set = self._get_article_keywords_set(kwargs)
        percent = self._calculate_percent_of_informal_words(article_keywords_set)
        return percent

    def percent_of_formal_words_in_post_image_text(self, **kwargs):
        post_image_text = self._get_post_image_text(kwargs)
        if post_image_text is not None:
            post_image_text_content = post_image_text.content
            word_set = get_words_by_content(post_image_text_content)
            percent = self._calculate_percent_of_formal_words(word_set)
            return percent
        return -1

    def percent_of_informal_words_in_post_image_text(self, **kwargs):
        post_image_text = self._get_post_image_text(kwargs)
        if post_image_text is not None:
            post_image_text_content = post_image_text.content
            word_set = get_words_by_content(post_image_text_content)
            percent = self._calculate_percent_of_informal_words(word_set)
            return percent
        return -1

    def _get_num_of_informal_words_by_item(self, item):
        words = self._get_words_by_given_item(item)
        num_of_formal_words = self._get_num_of_informal_words_from_language_dictionary(words)
        return num_of_formal_words

    def _get_num_of_informal_words_from_language_dictionary(self, words):
        total_num_of_words = len(words)
        num_of_formal_words = self._get_num_of_formal_words_from_language_dictionary(words)
        num_of_informal_words = total_num_of_words - num_of_formal_words
        return num_of_informal_words

    def _calculate_percent_of_formal_words(self, words):
        total_num_of_words = len(words)

        num_of_formal_words = self._get_num_of_formal_words_from_language_dictionary(words)

        percent = num_of_formal_words / total_num_of_words
        return percent

    def _calculate_percent_of_informal_words(self, words):
        total_num_of_words = len(words)

        num_of_formal_words = self._get_num_of_informal_words_from_language_dictionary(words)

        percent = num_of_formal_words / total_num_of_words
        return percent

    #-------------------------------------------------------------------------------------------

    def _get_words_post_title(self, kwargs):
        words = self._get_post_title(kwargs)
        return get_words_by_content(words)

    def _get_words_article_title(self, kwargs):
        words = self._get_article_title(kwargs)
        return get_words_by_content(words)

    def _get_words_article_description(self, kwargs):
        words = self._get_article_description(kwargs)
        return get_words_by_content(words)

# region_uppercase
    def number_of_uppercase_words_in_post_title(self, **kwargs):
        words = self._get_words_post_title(kwargs)
        return self._count_uppercase_words(words)

    def number_of_uppercase_words_in_article_title(self, **kwargs):
        words = self._get_words_article_title(kwargs)
        return self._count_uppercase_words(words)

    def number_of_uppercase_words_in_article_description(self, **kwargs):
        words = self._get_words_article_description(kwargs)
        return self._count_uppercase_words(words)

    def number_of_uppercase_words_in_article_captions(self, **kwargs):
        counter = 0
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return -1
        for capt in captions:
            words = get_words_by_content(capt)
            counter = counter + self._count_uppercase_words(words)
        return counter

    def number_of_uppercase_words_in_article_paragraphs(self, **kwargs):
        counter = 0
        paragraphs = self._get_article_paragraphs(kwargs)
        if paragraphs is None:
            return -1
        for paragraph in paragraphs:
            words = get_words_by_content(paragraph)
            counter = counter + self._count_uppercase_words(words)
        return counter

    def number_of_uppercase_words_in_article_keywords(self, **kwargs):
        words = self._get_article_keywords_set(kwargs)
        return self._count_uppercase_words(words)

    def number_of_uppercase_image_words(self, **kwargs):
        counter = 0
        images_text = self._get_post_image_text(kwargs)
        if images_text is None:
            return 0
        words = get_words_by_content(images_text.content)
        counter = counter + self._count_uppercase_words(words)
        return counter

    def are_all_post_title_words_uppercase(self, **kwargs):
        words = self._get_words_post_title(kwargs)
        num_of_uppercase = self._count_uppercase_words(words)
        num_of_words = len(words)
        return num_of_uppercase == num_of_words

    def are_all_article_title_words_uppercase(self, **kwargs):
        words = self._get_words_article_title(kwargs)
        num_of_upper_case = self._count_uppercase_words(words)
        num_of_words = len(words)
        return num_of_words == num_of_upper_case

    def are_all_article_keywords_uppercase(self, **kwargs):
        words = self._get_article_keywords_set(kwargs)
        num_of_upper_case = self._count_uppercase_words(words)
        num_of_words = len(words)
        return num_of_upper_case == num_of_words

    def are_all_article_captions_uppercase(self, **kwargs):
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return -1
        for capt in captions:
            words = get_words_by_content(capt)
            clean_words = self._clean_words(words)
            num_of_upper_case = self._count_uppercase_words(clean_words)
            num_of_words = len(clean_words)
        return num_of_words == num_of_upper_case

    def are_all_article_paragraph_uppercase(self, **kwargs):
        paragraphs = self._get_article_captions(kwargs)
        if paragraphs is None:
            return -1
        for paragraph in paragraphs:
            words = get_words_by_content(paragraph)
            clean_words = self._clean_words(words)
            num_of_upper_case = self._count_uppercase_words(clean_words)
            num_of_words = len(clean_words)
        return num_of_words == num_of_upper_case

    def are_all_article_image_text_uppercase(self, **kwargs):
        img_text = self._get_post_image_text(kwargs)
        if img_text is None:
            return 0
        words = img_text.content.split()
        num_of_uppercase = self._count_uppercase_words(words)
        num_of_words = len(words)
        return num_of_words == num_of_uppercase

    def _count_uppercase_words(self, words):
        counter = 0
        if words is None:
            return 0
        for word in words:
            if word.isupper():
                counter = counter + 1

        return counter

    def num_of_stopwords_in_post_title(self, **kwargs):
        words = self._get_words_post_title(kwargs)
        return self._count_stop_words(words)

    def num_of_stopwords_in_article_title(self, **kwargs):
        words = self._get_words_article_title(kwargs)
        return self._count_stop_words(words)

    def num_of_stopwords_in_article_description(self, **kwargs):
        words = self._get_words_article_description(kwargs)
        return self._count_stop_words(words)

    def num_of_stopwords_in_article_captions(self, **kwargs):
        counter = 0
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return 0
        for capt in captions:
            words = get_words_by_content(capt)
            counter = counter +self._count_stop_words(words)
        return counter

    def num_of_stopwords_in_article_keywords(self, **kwargs):
        words = self._get_article_keywords_set(kwargs)
        return self._count_stop_words(words)

    def num_of_stopwords_in_article_paragraphs(self, **kwargs):
        counter = 0
        paragraphs = self._get_article_paragraphs(kwargs)
        if paragraphs is None:
            return 0
        for paragraph in paragraphs:
            words = get_words_by_content(paragraph)
            counter = counter + self._count_stop_words(words)
        return counter

    def num_of_stopwords_in_image_text(self, **kwargs):
        counter = 0
        images_text = self._get_post_image_text(kwargs)
        if images_text is None:
            return 0
        words = get_words_by_content(images_text.content)
        counter = counter + self._count_stop_words(words)
        return counter

    def _count_stop_words(self, words):
        counter = 0
        for word in words:
            if word in self._stopwords_set:
                counter = counter + 1
        return counter


    def _load_stop_words(self):
        self._stopwords_set = stopwords.words('english')

    def _clean_word(self, word):
        return re.sub('[^a-zA-Z]+', '', word)

    #-----

    def contains_quotation_post_title(self, **kwargs):
        words = self._get_post_title(kwargs)
        return self._contains_quotations(words)

    def contains_quotation_article_title(self, **kwargs):
        words = self._get_article_title(kwargs)
        return self._contains_quotations(words)

    def contains_quotation_article_description(self, **kwargs):
        words = self._get_article_description(kwargs)
        return self._contains_quotations(words)

    def contains_quotation_article_captions(self, **kwargs):
        counter = 0
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return 0
        for capt in captions:
            text = get_words_by_content(capt)
            for words in text:
                if self._contains_quotations(words):
                    counter = counter + 1
        return counter

    def contains_quotation_article_keywords(self, **kwargs):
        words = self._get_article_keywords_set(kwargs)
        if words is None:
            return 0
        counter = 0
        for word in words:
            if self._contains_quotations(word):
                counter = counter + 1
        return counter

    def contains_quotation_article_paragraph(self, **kwargs):
        counter = 0
        paragraphs = self._get_article_paragraphs(kwargs)
        if paragraphs is None:
            return 0
        for paragraph in paragraphs:
            text = get_words_by_content(paragraph)
            for words in [_f for _f in text if _f]:
                if self._contains_quotations(words):
                    counter = counter +1
        return counter

    def contains_quotation_post_image(self, **kwargs):
        counter = 0
        images_text = self._get_post_image_text(kwargs)
        if images_text is None:
            return 0
        text = get_words_by_content(images_text.content)
        for words in [_f for _f in text if _f]:
            if self._contains_quotations(words):
                counter = counter + 1
        return counter

    def _contains_quotations(self, text):
        if text is None:
            return 0
        start_index = text.find("\"")
        if start_index != -1:  # i.e. if the first quote was found
            endIndex = text.find('\"', start_index + 1)
            if start_index != -1 and endIndex != -1:  # i.e. both quotes were found
                return 1
            else:
                return 0
        return 0

    def _get_meaning_set(self, sentence):
        if sentence is None or sentence == '':
            return None
        tags = word_tokenize(sentence)
        proccessed_tags = nltk.pos_tag(tags)
        simplified_tags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in proccessed_tags]
        return simplified_tags

    def _num_of_nouns_in_sentence(self, sentence):
        meaning_dict = self._get_meaning_set(sentence)
        if meaning_dict is None:
            return 0
        counter = 0
        for word in meaning_dict:
            if word[1] == 'NOUN':
                counter = counter+1
        return counter

    def num_of_nouns_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        return self._num_of_nouns_in_sentence(post_title)

    def num_of_nouns_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        return self._num_of_nouns_in_sentence(article_title)

    def num_of_nouns_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        return self._num_of_nouns_in_sentence(article_description)

    def num_of_nouns_in_article_paragraphs(self, **kwargs):
        paragraphs = self._get_article_paragraphs(kwargs)
        if paragraphs is None:
            return -1
        counter = 0
        for paragraph in paragraphs:
            counter = counter + self._num_of_nouns_in_sentence(paragraph)

    def num_of_nouns_in_article_captions(self, **kwargs):
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return -1
        counter = 0
        for caption in captions:
            counter = counter + self._num_of_nouns_in_sentence(caption)

    def num_of_nouns_in_article_keywords(self, **kwargs):
        keywords = self._get_article_keywords_string(kwargs)
        if keywords is None:
            return -1
        return self._num_of_nouns_in_sentence(keywords)

    def num_of_nouns_in_article_image(self, **kwargs):
        post_image = self._get_post_image_text(kwargs)
        if post_image is None:
            return -1
        words = post_image.content
        return self._num_of_nouns_in_sentence(words)
    #--verbs
    def _num_of_verbs_in_sentence(self, sentence):
        meaning_dict = self._get_meaning_set(sentence)
        if meaning_dict is None:
            return 0
        counter = 0
        for word in meaning_dict:
            if word[1] == 'VERB':
                counter = counter+1
        return counter

    def num_of_verbs_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        return self._num_of_verbs_in_sentence(post_title)

    def num_of_verbs_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        return self._num_of_verbs_in_sentence(article_title)

    def num_of_verbs_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        return self._num_of_verbs_in_sentence(article_description)

    def num_of_verbs_in_article_paragraphs(self, **kwargs):
        paragraphs = self._get_article_paragraphs(kwargs)
        if paragraphs is None:
            return -1
        counter = 0
        for paragraph in paragraphs:
            counter = counter + self._num_of_verbs_in_sentence(paragraph)

    def num_of_verbs_in_article_captions(self, **kwargs):
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return -1
        counter = 0
        for caption in captions:
            counter = counter + self._num_of_verbs_in_sentence(caption)

    def num_of_verbs_in_article_keywords(self, **kwargs):
        keywords = self._get_article_keywords_string(kwargs)
        if keywords is None:
            return -1
        return self._num_of_verbs_in_sentence(keywords)

    def num_of_verbs_in_article_image(self, **kwargs):
        post_image = self._get_post_image_text(kwargs)
        if post_image is None:
            return -1
        words = post_image.content
        return self._num_of_verbs_in_sentence(words)

    #--- adj
    def _num_of_adjs_in_sentence(self, sentence):
        meaning_dict = self._get_meaning_set(sentence)
        if meaning_dict is None:
            return 0
        counter = 0
        for word in meaning_dict:
            if word[1] == 'ADJ':
                counter = counter + 1
        return counter

    def num_of_adjs_in_post_title(self, **kwargs):
        post_title = self._get_post_title(kwargs)
        return self._num_of_adjs_in_sentence(post_title)

    def num_of_adjs_in_article_title(self, **kwargs):
        article_title = self._get_article_title(kwargs)
        return self._num_of_adjs_in_sentence(article_title)

    def num_of_adjs_in_article_description(self, **kwargs):
        article_description = self._get_article_description(kwargs)
        return self._num_of_adjs_in_sentence(article_description)

    def num_of_adjs_in_article_paragraphs(self, **kwargs):
        paragraphs = self._get_article_paragraphs(kwargs)
        if paragraphs is None:
            return -1
        counter = 0
        for paragraph in paragraphs:
            counter = counter + self._num_of_adjs_in_sentence(paragraph)

    def num_of_adjs_in_article_captions(self, **kwargs):
        captions = self._get_article_captions(kwargs)
        if captions is None:
            return -1
        counter = 0
        for caption in captions:
            counter = counter + self._num_of_adjs_in_sentence(caption)

    def num_of_adjs_in_article_keywords(self, **kwargs):
        keywords = self._get_article_keywords_string(kwargs)
        if keywords is None:
            return -1
        return self._num_of_adjs_in_sentence(keywords)

    def num_of_adjs_in_article_image(self, **kwargs):
        post_image = self._get_post_image_text(kwargs)
        if post_image is None:
            return -1
        words = post_image.content
        return self._num_of_adjs_in_sentence(words)