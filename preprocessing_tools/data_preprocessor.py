# Created by jorgeaug at 10/04/2016
import os
import re
from nltk.stem.snowball import GermanStemmer, EnglishStemmer
from commons.consts import Language
from bs4 import BeautifulSoup
from preprocessing_tools.abstract_controller import AbstractController
import time
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

class Preprocessor(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._remove_stopwords    = self._config_parser.eval(self.__class__.__name__, "remove_stopwords")
        self._apply_stemming      = self._config_parser.eval(self.__class__.__name__, "apply_stemming")
        self._stopwords_file = self._config_parser.get(self.__class__.__name__, "stopwords_file")
        self._stemming_lang = self._config_parser.get(self.__class__.__name__, "stemming_language")
        self._is_preprocess_posts = self._config_parser.eval(self.__class__.__name__, "is_preprocess_posts")
        self._is_preprocess_authors = self._config_parser.eval(self.__class__.__name__, "is_preprocess_authors")
        self._max_objects_without_saving = self._config_parser.eval(self.__class__.__name__, "max_objects_without_saving")

        pattern_regex = "[A-Za-z']+".decode('utf-8')
        self.compiled_regex = re.compile(pattern_regex)

    def set_up(self):
        pass

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("Executing " + self.__class__.__name__)

        if self._remove_stopwords:
            self.load_stop_words()

        if self._apply_stemming:
            self.load_stemmer()

        if self._is_preprocess_posts:
            posts = self._db.get_posts_filtered_by_domain(self._domain)
            self._preprocess_posts(posts)
        if self._proprocess_authors:
            authors = self._db.get_authors(self._domain)
            self._proprocess_authors(authors)

        end_time = time.time()
        diff_time = end_time - start_time
        logging.info('execute finished in '+ str(diff_time) +' seconds')

    def remove_html_tags(self, text):
        text = BeautifulSoup(text, 'lxml').get_text()
        return text

    def load_stemmer(self):
        self._stemmer = None
        if self._stemming_lang == Language.GERMAN:
            self._stemmer = GermanStemmer()
        else:
            self._stemmer = EnglishStemmer()

    def load_stop_words(self):
        #file_path = self._config_parser.get(self.__class__.__name__, "stopwords_file")
        spaces = "\r\n\t "
        if os.path.exists(self._stopwords_file):
            with open(self._stopwords_file) as file:
                self._stopwords = set((line.strip(spaces) for line in file.readlines()))

    def stem_text(self, text):
        tokens = [_ for _ in text.split(" ") if len(_) > 0]
        words = [self._stemmer.stem(word) for word in tokens]
        content = " ".join(words)
        return content

    def remove_stopwords(self, content):
        words = content.split()
        words = [word for word in words if word not in self._stopwords]
        return " ".join(words)

    def _preprocess_posts(self, posts):
        preprocessed_posts = []
        number_posts = len(posts)
        i = 0
        for post in posts:
            i += 1
            if i % self._max_objects_without_saving == 0:
                logging.info("processing author " + str(i) + " of " + str(number_posts))
            if post.content is not None:
                post.content = str(post.content).lower()
                post.content = self.remove_html_tags(post.content)
                if self._remove_stopwords:
                    post.content = self.remove_stopwords(post.content)
                if self._apply_stemming:
                    post.content = self.stem_text(post.content)

            if post.title is not None:
                post.title = str(post.title).lower()
                post.title = self.remove_html_tags(post.title)
                if self._remove_stopwords:
                    post.title = self.remove_stopwords(post.title)
                if self._apply_stemming:
                    post.title = self.stem_text(post.title)

            preprocessed_posts.append(post)
            if len(preprocessed_posts) == self._max_objects_without_saving:
                self._db.addPosts(preprocessed_posts)
                preprocessed_posts = []

        self._db.addPosts(preprocessed_posts)

    def _proprocess_authors(self, authors):
        preprocessed = []
        number_posts = len(authors)
        i = 0
        for author in authors:
            i += 1
            if i % self._max_objects_without_saving == 0:
                logging.info("processing author " + str(i) + " of " + str(number_posts))
            if author.description is not None:
                original_description = author.description
                original_description = str(original_description).lower()
                author.description = self.remove_html_tags(original_description)
                if self._remove_stopwords:
                    author.description = self.remove_stopwords(author.description)
                if self._apply_stemming:
                    author.description = self.stem_text(author.description)
            preprocessed.append(author)
            if len(preprocessed) == self._max_objects_without_saving:
                self._db.addAuthors(preprocessed)
                preprocessed = []

        self._db.addAuthors(preprocessed)