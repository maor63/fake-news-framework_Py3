
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd
import numpy as np
import re
import itertools
from collections import Counter
from nltk.corpus import stopwords

# import networkx as nx
# import urllib2
# from commons.method_executor import Method_Executor
# from commons.commons import count_down_time

__author__ = "Aviad Elyashar"



def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    returned_words = []
    for word in words:
        if word not in stop_words:
            returned_words.append(word)
    return returned_words


def find_keywords_in_given_items(items, verified_keywords_set):
    is_found = 0
    found_word = ""
    for item in items:
        if item in verified_keywords_set:
            is_found = 1
            found_word = item
            break
    return is_found, found_word



class HealthcareWorkerDetector(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

        self._input_path = self._config_parser.eval(self.__class__.__name__, "input_path")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._keywords_file_name = self._config_parser.eval(self.__class__.__name__, "keywords_file_name")

    def execute(self, window_start=None):

        keywords_df = pd.read_csv(self._input_path + self._keywords_file_name)
        print("strating get_all_authors")
        authors = self._db.get_all_authors()
        print("finished get_all_authors")

        print("creating author_osn_id_author_dict")
        self._author_osn_id_author_dict = {author.author_osn_id : author for author in authors}

        print("get descriptions and full names")
        author_descr_tuples = self._db.get_description_and_full_names_for_authors()

        author_descr_df = pd.DataFrame(author_descr_tuples, columns=['author_guid', 'author_osn_id', 'author_screen_name',
                                                   'author_full_name', 'description'])

        print("finishing get descriptions and full names")

        #author_descr_df = author_descr_df.head(100)
        # clean_description

        author_descr_df = author_descr_df.replace(np.nan, '', regex=True)


        author_descr_df = self._low_case_and_remove_stop_words_in_description(author_descr_df)

        author_descr_df = self._words_detected_in_description(author_descr_df)

        author_descr_df = self._low_case_and_remove_stop_words_in_full_name(author_descr_df)

        author_descr_df = self._words_detected_in_full_name(author_descr_df)


        author_descr_df["Is_keyword_detected_in_both_desc_and_full_name"] = author_descr_df[
                                                                                  "Is_keyword_detected_in_full_name"] + \
                                                                            author_descr_df[
                                                                                  "Is_keyword_detected_in_description"]

        self._calculate_statistics(author_descr_df)

        osn_ids = self._get_detected_author_ids(author_descr_df)

        authors_to_update = []
        for osn_id in osn_ids:
            author = self._author_osn_id_author_dict[osn_id]
            author.author_type = "Healthcare_Worker_Auto"
            authors_to_update.append(author)

        print("Authors detected as healthcare workers: {0}".format(len(authors_to_update)))
        self._db.addPosts(authors_to_update)


        print("Done!!")




    def _calculate_statistics(self, author_descr_df):
        results = []

        author_descr_group_by_df = author_descr_df.groupby(['Is_keyword_detected_in_description']).count()
        not_detected_by_description = author_descr_group_by_df['author_guid'][0]
        res_tuple = ("not_detected_by_description", not_detected_by_description)
        results.append(res_tuple)

        detected_by_description = author_descr_group_by_df['author_guid'][1]
        res_tuple = ("detected_by_description", detected_by_description)
        results.append(res_tuple)

        author_full_name_group_by_df = author_descr_df.groupby(['Is_keyword_detected_in_full_name']).count()
        not_detected_by_full_name = author_full_name_group_by_df['author_guid'][0]
        res_tuple = ("not_detected_by_full_name", not_detected_by_full_name)
        results.append(res_tuple)

        detected_by_full_name = author_full_name_group_by_df['author_guid'][1]
        res_tuple = ("detected_by_full_name", detected_by_full_name)
        results.append(res_tuple)

        author_descr_and_full_name_group_by_df = author_descr_df.groupby(
            ['Is_keyword_detected_in_both_desc_and_full_name']).count()
        not_detected_by_both = author_descr_and_full_name_group_by_df['author_guid'][0]
        res_tuple = ("not_detected_by_both", not_detected_by_both)
        results.append(res_tuple)

        detected_by_only_one_of_them = author_descr_and_full_name_group_by_df['author_guid'][1]
        res_tuple = ("detected_by_only_one_of_them", detected_by_only_one_of_them)
        results.append(res_tuple)

        detected_by_descr_and_full_name = author_descr_and_full_name_group_by_df['author_guid'][2]
        res_tuple = ("detected_by_descr_and_full_name", detected_by_descr_and_full_name)
        results.append(res_tuple)

        detected_by_both = detected_by_only_one_of_them + detected_by_descr_and_full_name
        res_tuple = ("detected_by_both", detected_by_both)
        results.append(res_tuple)

        results_df = pd.DataFrame(results, columns=['Key', 'Value'])
        results_df.to_csv(self._output_path + "Healthcare_Workers_Detection_Statistics.csv", index=False)

    def _low_case_and_remove_stop_words_in_description(self, author_descr_df):
        print("starting _low_case_and_remove_stop_words_in_description")

        author_descr_df["lower_description"] = author_descr_df["description"].apply(lambda x: x.lower())
        author_descr_df["lower_description"] = author_descr_df["lower_description"].apply(
            lambda x: re.sub(r'\W+', ' ', x))
        # author_descr_df["lower_description"] = author_descr_df["lower_description"].apply(lambda x: re.sub(r'\n', ' ', x))
        author_descr_df["description_words"] = author_descr_df["lower_description"].apply(
            lambda x: ''.join(x).split(' '))

        author_descr_df["no_stopwords_description_words"] = author_descr_df["description_words"].apply(
            lambda x: remove_stopwords(x))

        print("finishing _low_case_and_remove_stop_words_in_description")

        return author_descr_df


    def _words_detected_in_description(self, author_descr_df):
        print("starting _words_detected_in_description")
        # detect by description
        verified_keywords_df = keywords_df[keywords_df["Is a good keywords?"] == 1]
        verified_keywords_series = verified_keywords_df["Word"]
        verified_keywords = verified_keywords_series.tolist()
        verified_keywords_set = set(verified_keywords)

        author_descr_df["Is_keyword_detected_in_description"], author_descr_df[
            "First_found_keyword_in_description"] = list(zip(*author_descr_df["no_stopwords_description_words"].apply(
            lambda x: find_keywords_in_given_items(x, verified_keywords_set))))

        print("finishing _words_detected_in_description")
        return author_descr_df

    def _low_case_and_remove_stop_words_in_full_name(self, author_descr_df):
        print("starting _low_case_and_remove_stop_words_in_full_name")
        author_descr_df["lower_author_full_name"] = author_descr_df["author_full_name"].apply(lambda x: x.lower())
        author_descr_df["lower_author_full_name"] = author_descr_df["lower_author_full_name"].apply(
            lambda x: re.sub(r'\W+', ' ', x))
        # selected_users_df["lower_author_full_name"] = selected_users_df["lower_author_full_name"].apply(lambda x: re.sub(r'\n', ' ', x))
        author_descr_df["full_name_words"] = author_descr_df["lower_author_full_name"].apply(
            lambda x: ''.join(x).split(' '))

        author_descr_df["no_stopwords_full_name_words"] = author_descr_df["full_name_words"].apply(
            lambda x: remove_stopwords(x))

        print("finishing _low_case_and_remove_stop_words_in_full_name")
        return author_descr_df

    def _words_detected_in_full_name(self, author_descr_df):
        print("starting _words_detected_in_full_name")
        author_descr_df["Is_keyword_detected_in_full_name"], author_descr_df[
            "First_found_keyword_in_full_name"] = list(zip(*author_descr_df["no_stopwords_full_name_words"].apply(
            lambda x: find_keywords_in_given_items(x, verified_keywords_set))))

        print("finishing _words_detected_in_full_name")
        return author_descr_df

    def _get_detected_author_ids(self, author_descr_df):
        detected_by_descr_or_full_name_df = author_descr_df[
            author_descr_df['Is_keyword_detected_in_both_desc_and_full_name'] == 1]
        descr_or_full_name_osn_id_series = detected_by_descr_or_full_name_df["author_osn_id"]
        descr_or_full_name_osn_ids = descr_or_full_name_osn_id_series.tolist()

        detected_by_descr_and_full_name_df = author_descr_df[
            author_descr_df['Is_keyword_detected_in_both_desc_and_full_name'] == 2]
        descr_and_full_name_osn_id_series = detected_by_descr_and_full_name_df["author_osn_id"]
        descr_and_full_name_osn_ids = descr_and_full_name_osn_id_series.tolist()

        osn_ids = list(set(descr_or_full_name_osn_ids + descr_and_full_name_osn_ids))
