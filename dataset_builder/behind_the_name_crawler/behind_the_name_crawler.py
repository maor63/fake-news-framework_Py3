
import pandas as pd
import networkx as nx
import urllib.request, urllib.error, urllib.parse
from commons.method_executor import Method_Executor
from commons.commons import count_down_time


__author__ = "Aviad Elyashar"


class BehindTheNameCrawler(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_directory_path = self._config_parser.eval(self.__class__.__name__, "input_directory_path")
        self._target_file_name = self._config_parser.eval(self.__class__.__name__, "target_file_name")
        self._output_directory_path = self._config_parser.eval(self.__class__.__name__, "output_directory_path")
        self._results_file_name = self._config_parser.eval(self.__class__.__name__, "results_file_name")

        #self._api_key = "av389484446"
        #self._api_key = "av817316338"
        self._max_requests_per_seconds = 5
        self._max_requests_per_hour = 1000

        self._api_keys = ['be856290695', 'te779995984', 'me331166229', 'an979531289', 'le287963274',
                          'am844427433', 'an059372313', 'gi011368962', 'ga979452073', 'ca307261595']

        self._api_key = "av817316338"
        self._current_selector = 6
        self._original_name_synonym_df = pd.DataFrame()
        self._problematic_names_df = pd.DataFrame()
        self._problematic_results_file_name = "problematic_names.csv"

        # be856290695 - Benny
        # te779995984 - tedaringthon
        # meggiewill@walla.com - me331166229
        # annakiril@walla.com - an979531289
        # avavilevi@walla.com - le287963274
        # amirabuzavgo@walla.com - am844427433
        # andrea_rossi@walla.com -an059372313
        # giuliaesposito@walla.com - gi011368962
        # garciaantonio@walla.com - ga979452073
        # camilarod@walla.com - ca307261595



    def crawl_names_by_child_ancestor_aggregated_file(self):
        names_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        names_df = names_df.rename(index=str, columns={"Child_First_Name": "source",
                                                       "Ancestor_First_Name": "target",
                                                       "count": "weight"})
        names_graph = nx.from_pandas_edgelist(names_df, edge_attr=True)
        names = list(names_graph.nodes)
        sorted_names = sorted(names)
        sorted_names = [sorted_name for sorted_name in sorted_names if len(sorted_name) > 1]
        sorted_names_df = pd.DataFrame(sorted_names, columns=['Name:'])
        sorted_names_df.to_csv(self._output_directory_path + "sorted_names.csv", index=False)

        self._crawl_names_and_export_results(sorted_names)

        print("Done!!!!!")

    def crawl_names_by_child_ancestor_aggregated_file2(self):
        names_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        child_first_name_series = names_df["Child_First_Name"]
        child_first_names = child_first_name_series.tolist()

        ancestor_first_name_series = names_df["Ancestor_First_Name"]
        ancestor_first_names = ancestor_first_name_series.tolist()

        all_names = child_first_names + ancestor_first_names
        all_names = list(set(all_names))
        sorted_names = [name for name in all_names if len(name) > 1]
        sorted_names = sorted(sorted_names)
        sorted_names_df = pd.DataFrame(sorted_names, columns=['Name:'])
        sorted_names_df.to_csv(self._output_directory_path + "sorted_names_updated2.csv", index=False)


        print("Num of names to check: {0}".format(len(sorted_names)))

        self._crawl_names_and_export_results(sorted_names)

        print("Done!!!!!")

    def crawl_names_by_name_file(self):
        names_df = pd.read_csv(self._input_directory_path + self._target_file_name)
        targeted_names_series = names_df['Name']
        targeted_names = targeted_names_series.tolist()
        targeted_names = sorted(targeted_names)

        #self._crawl_names_and_export_results(targeted_names)
        self._crawl_names_and_move_between_api_keys_export_results(targeted_names)

        print("Done!!!!!")


    def _export_results(self, results):
        results_df = pd.DataFrame(results, columns=['Name', 'Synonym'])
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)

    def _crawl_names_and_export_results(self, names):
        results = []
        problematic_names = []
        total_requests = 0
        requests_in_seconds = 0
        for i, name in enumerate(names):
            msg = "\r {0}/{1} - synonyms for {2}".format(i, len(names), name)
            print(msg, end='')
            try:
                url = "https://www.behindthename.com/api/related.json?name={0}&key={1}".format(name, self._api_key)
                response = urllib.request.urlopen(url)
                json = response.read()
                synonyms_dict = eval(json)
                if 'names' in synonyms_dict:
                    synonyms = synonyms_dict['names']
                    for synonym in synonyms:
                        synonym_tuple = (name, synonym)
                        results.append(synonym_tuple)
                else:
                    problem = (name, "name could not be found")
                    problematic_names.append(problem)

                total_requests += 1
                requests_in_seconds += 1

                if requests_in_seconds > self._max_requests_per_seconds:
                    count_down_time(2)
                    requests_in_seconds = 0

                if total_requests > self._max_requests_per_hour:
                    count_down_time(3660)
                    total_requests = 0





            except:
                print("Problematic name:{0}".format(name))
                print("Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")

                problem = (name, "Unknown Error")
                problematic_names.append(problem)
        
        self._export_results(results)

        if len(problematic_names) > 1:
            results_df = pd.DataFrame(problematic_names, columns=['Problematic_Name', "Error"])
            results_df.to_csv(self._output_directory_path + "problematic_names.csv", index=False)

    def _export_results_when_finishing(self, results):
        results_df = pd.DataFrame(results, columns=['Name', 'Synonym'])
        results_df.to_csv(self._output_directory_path + self._results_file_name, index=False)


    def _save_results_in_the_middle(self, results, united_results_df, results_file_name):
        middle_results_df = pd.DataFrame(results, columns=['Name', 'Synonym'])
        united_results_df = united_results_df.append(middle_results_df)
        united_results_df.to_csv(self._output_directory_path + results_file_name, index=False)
        return united_results_df

    # def save_results_in_the_middle(self, results):
    #     middle_results_df = pd.DataFrame(results, columns=['Name', 'Synonym'])
    #     self._original_name_synonym_df = self._original_name_synonym_df.append(middle_results_df)
    #     self._original_name_synonym_df.to_csv(self._output_directory_path + "all_names_synonyms.csv", index=False)
    #
    # def _save_problematic_names_in_the_middle(self, problematic_names, problematic_names_df):
    #     middle_results_problematic_df = pd.DataFrame(problematic_names,
    #                                                  columns=['Problematic_Name', "Error"])
    #     problematic_names_df = problematic_names_df.append(middle_results_problematic_df)
    #     problematic_names_df.to_csv(self._output_directory_path + "problematic_names.csv",
    #                                 index=False)

    def _switch_api_key(self):
        # Switching api_key instead of waiting
        previous_api_key = self._api_key

        self._current_selector += 1
        if self._current_selector == 10:
            self._current_selector = 0

        self._api_key = self._api_keys[self._current_selector]
        print("Switching api_key from - to:".format(previous_api_key, self._api_key))

    def _save_middle_results_and_switch_api_key(self, results, problematic_names):
        self._original_name_synonym_df = self._save_results_in_the_middle(results, self._original_name_synonym_df,
                                                                          self._results_file_name)
        self._problematic_names_df = self._save_results_in_the_middle(problematic_names, self._problematic_names_df,
                                                                      self._problematic_results_file_name)

        # count_down_time(3660)

        # Switching api_key instead of waiting
        self._switch_api_key()


    def _crawl_names_and_move_between_api_keys_export_results(self, names):
        self._api_key = self._api_keys[self._current_selector]
        print("Current API key:{0}".format(self._api_key))

        results = []
        problematic_names = []
        total_requests = 1
        requests_in_seconds = 1
        for i, name in enumerate(names):
            msg = "\r {0}/{1} - synonyms for {2}".format(i, len(names), name)
            print(msg, end='')
            try:
                if requests_in_seconds > self._max_requests_per_seconds:
                    count_down_time(2)
                    requests_in_seconds = 1

                if total_requests > self._max_requests_per_hour:
                    self._save_middle_results_and_switch_api_key(results, problematic_names)
                    results = []
                    problematic_names = []

                    total_requests = 1

                # url = "https://www.behindthename.com/api/related.json?name={0}&usage=eng&key={1}".format(name, self._api_key)
                url = "https://www.behindthename.com/api/related.json?name={0}&key={1}".format(name, self._api_key)
                response = urllib.request.urlopen(url)
                json = response.read()
                synonyms_dict = eval(json)
                if 'names' in synonyms_dict:
                    synonyms = synonyms_dict['names']
                    for synonym in synonyms:
                        synonym_tuple = (name, synonym)
                        results.append(synonym_tuple)
                else:
                    problem = (name, "name could not be found")
                    problematic_names.append(problem)


                total_requests += 1
                requests_in_seconds += 1

            except:
                print("Problematic name:{0}".format(name))
                print("Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")

                problem = (name, "Unknown Error")
                problematic_names.append(problem)

                total_requests += 1
                requests_in_seconds += 1

        # self._export_results(results)
        self._save_results_in_the_middle(results, self._original_name_synonym_df, self._results_file_name)

        if len(problematic_names) > 1:
            self._save_results_in_the_middle(problematic_names, self._problematic_names_df, self._problematic_results_file_name)
            # middle_results_problematic_df = pd.DataFrame(problematic_names, columns=['Problematic_Name', "Error"])
            # problematic_names_df = problematic_names_df.append(middle_results_problematic_df)
            # problematic_names_df.to_csv(self._output_directory_path + "problematic_names.csv",
            #                             index=False)
            # results_df = pd.DataFrame(problematic_names, columns=['Problematic_Name', "Error"])
            # results_df.to_csv(self._output_directory_path + "problematic_names.csv", index=False)