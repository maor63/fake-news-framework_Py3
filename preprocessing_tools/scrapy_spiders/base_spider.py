import os
import string
from urllib.request import Request, urlopen

import nltk
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from scrapy import Spider
from scrapy.cmdline import execute
from commons.commons import *

class BaseSpider(Spider):
    @classmethod
    def set_output_path(cls, output_path):
        cls.custom_settings = cls.get_settings(cls.name, output_path)

    @staticmethod
    def get_settings(name, output_path=''):
        return {
            'FEED_FORMAT': 'csv',
            'FEED_URI': os.path.join(output_path, '{}_claims.csv'.format(name)),
        }

    def open_fact_check_page(self, url):
        request = Request(url, headers={'User-Agent': 'Chclean_site_tagsrome/70.0.3538.102'})
        return BeautifulSoup(urlopen(request), "html.parser")

    def get_optional_tags(self, current_tags, claim_tags):
        optional_tags = []
        for tag in claim_tags:
            if tag.lower() not in current_tags.lower():
                optional_tags.append(tag)
        if current_tags:
            current_tags += ','
        return current_tags + ','.join(optional_tags)

    def extract_tags(self, claim):
        filtered_sentence = []
        tags = []
        # claim = claim.lower()
        stop_words = set(stopwords.words('english'))
        not_allowed_input = set(string.punctuation)
        for w in nltk.word_tokenize(claim):
            if w not in stop_words and all(c not in w for c in not_allowed_input) and len(w) != 1:
                filtered_sentence.append(w)
        lemmatizer = WordNetLemmatizer()
        filtered_sentence = nltk.pos_tag(filtered_sentence)
        for word_and_pos_tag in filtered_sentence:
            if word_and_pos_tag[1].startswith('NN'):
                word_lemmatize = lemmatizer.lemmatize(word_and_pos_tag[0])
                if not any(word_lemmatize.lower() in tag.lower() for tag in tags):
                    tags.append(word_lemmatize)
        return tags

    def replace_suffix_in_date(self, date):
        all_suffix = ["th", "rd", "nd", "st"]
        for suffix in all_suffix:
            date = date.replace(suffix, '')
        return date

    def clean_site_tags(self, site_tags):
        new_tags = []
        for tag in site_tags:
            new_tag = ''
            for char in tag:
                if char.isalpha() or char.isdigit() or char.isspace():
                    new_tag += char
            if new_tag != '':
                new_tags.append(new_tag)
        return ','.join(new_tags)

    def get_page_soup(self, response):
        return BeautifulSoup(response.text, "html.parser")

    def extract_articles(self, top_pages=None):
        if top_pages:
            execute(('scrapy runspider %s.py -o %s_claims.csv -t csv -a limit=%s' % (
                self.name, self.name, top_pages)).split())
        else:
            execute(('scrapy runspider %s.py -o %s_claims.csv -t csv' % (self.name, self.name)).split())

    def export_row(self, domain, title, claim, description, url, verdict_date, tags, category, label, image_src):
        return {'domain': domain,
                'title': self.fix_text_for_csv(title),
                'claim': self.fix_text_for_csv(claim),
                'description': self.fix_text_for_csv(description),
                'url': url,
                'verdict_date': date_to_str(verdict_date),
                'keywords': tags.replace(',', '||'),
                'main_category': category.replace(',', '||'),
                'secondary_category': '',
                'verdict': label,
                'image_src': image_src}

    def fix_text_for_csv(self, text):
        text = text[:10000]
        if ',' in text:
            text = '"{}"'.format(text.replace('"', ''))
            text = ','.join(text.split(','))
        text = text.replace('\n', ' ').replace('\r', '')
        text = remove_punctuation_chars(text)
        return text

    def get_pages_to_crawl(self, total_pages):
        if self.settings.getbool('UPDATE'):
            total_pages = 2
        return total_pages
