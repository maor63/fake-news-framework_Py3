
import pandas as pd
from commons.method_executor import Method_Executor
from commons.commons import count_down_time
import requests
import random
from bs4 import BeautifulSoup

__author__ = "Aviad Elyashar"


class BehindTheNameLastNameScraper(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_directory_path = self._config_parser.eval(self.__class__.__name__, "input_directory_path")
        self._target_file_name = self._config_parser.eval(self.__class__.__name__, "target_file_name")
        self._output_directory_path = self._config_parser.eval(self.__class__.__name__, "output_directory_path")
        self._results_file_name = self._config_parser.eval(self.__class__.__name__, "results_file_name")

        self._total_requests = 0
        self._max_requests_per_hour = 1000
        self._name_related_name_tuples = []
        self._name_related_name_df = pd.DataFrame()
        self._name_not_found_tuples = []
        self._name_not_found_df = pd.DataFrame()



    def crawl_related_names_for_last_names(self):
        #last_names = ["Aaaa", "Williams", "Cohen", "Nazim", "Robinson"]

        last_names = self._get_last_names_from_file()

        self._total_requests = 1

        for i, last_name in enumerate(last_names):
            msg = "\r {0}/{1} - synonyms for {2}".format(i, len(last_names), last_name)
            print(msg, end='')

            if self._total_requests > self._max_requests_per_hour:
                self._save_results_and_go_to_sleep()

            try:
                self._send_request_and_save_result(last_name)

            except Exception as e:
                print("Errorrrrrrrrrrrrrrrrrrrrrrrrrr")

                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)


        self._save_results_in_the_middle(self._name_related_name_tuples, self._name_related_name_df, self._results_file_name)
        self._save_results_in_the_middle(self._name_not_found_tuples, self._name_not_found_df,
                                         "Synonyms_Not_Found_For_Last_Names.csv")

        # name_related_name_df = pd.DataFrame(self._name_related_name_tuples, columns=['Last Name', 'Synonym', 'Language', 'Type'])
        # name_related_name_df.to_csv(self._output_directory_path + self._results_file_name, encoding='utf-8', index=False)
        #
        # related_name_not_found_df = pd.DataFrame(self._name_not_found_tuples, columns=['Last Name', 'Reason'])
        # related_name_not_found_df.to_csv(self._output_directory_path + "Synonyms_Not_Found_For_Last_Names.csv", encoding='utf-8', index=False)


    def _extract_related_name_from_html(self, last_name, response):
        # sleep 30 seconds
        count_down_time(random.randint(10, 30))

        # Parse HTML and save to BeautifulSoup object
        soup = BeautifulSoup(response.text, "html.parser")
        headlines_h3 = soup.findAll("h3", {"class": "related"})
        related_block_divs = soup.findAll("div", {"class": "related-block"})

        for i, headline in enumerate(headlines_h3):
            related_name_headline = headlines_h3[i]
            headline_contents = related_name_headline.contents
            headline = headline_contents[0]

            related_block_div = related_block_divs[i]
            related_block_div_contents = related_block_div.contents

            related_div_contents = [element for element in related_block_div_contents if hasattr(element, 'contents')]
            for related_name_div in related_div_contents:
                language_tag = related_name_div.findAll('b')
                language = language_tag[0].text

                related_name_tags = related_name_div.findAll('a')
                related_names = [related_name_tag.text for related_name_tag in related_name_tags]

                for related_name in related_names:
                    related_name_tuple = (last_name, related_name, language, headline)
                    self._name_related_name_tuples.append(related_name_tuple)


    def _save_results_in_the_middle(self, results, united_results_df, results_file_name):
        middle_results_df = pd.DataFrame(results, columns=['Last Name', 'Synonym', 'Language', 'Type'])
        united_results_df = united_results_df.append(middle_results_df)
        united_results_df.to_csv(self._output_directory_path + results_file_name, encoding='utf-8', index=False)
        return united_results_df

    def _save_results_and_go_to_sleep(self):
        self._save_results_in_the_middle(self._name_related_name_tuples, self._name_related_name_df,
                                         self._results_file_name)
        self._total_requests = 0
        self._name_related_name_tuples = []

        self._save_results_in_the_middle(self._name_not_found_tuples, self._name_not_found_df,
                                         "Synonyms_Not_Found_For_Last_Names.csv")
        self._name_not_found_tuples = []

        count_down_time(10)

    def _get_last_names_from_file(self):
        last_names_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        last_names_series = last_names_df['Name']
        last_names = last_names_series.tolist()
        last_names = sorted(last_names)
        return last_names

    def _send_request_and_save_result(self, last_name):
        # Set the URL you want to webscrape from
        url = "https://surnames.behindthename.com/name/{0}/related".format(last_name)

        # Connect to the URL
        response = requests.get(url)

        # returns good answer - the name exists in the website
        if response.status_code == 200:
            self._extract_related_name_from_html(last_name, response)

        else:
            name_not_found_tuple = (last_name, "Name not found", "No", "No")
            self._name_not_found_tuples.append(name_not_found_tuple)

        self._total_requests += 1
