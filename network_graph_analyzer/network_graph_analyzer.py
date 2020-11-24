

from commons.method_executor import Method_Executor
import os
import graphlab as gl
import re
import pandas as pd
import urllib.request, urllib.parse, urllib.error
from BeautifulSoup import BeautifulSoup
import tarfile
import threading
import logging

__author__ = "Aviad Elyashar"

class NetworkGraphAnalyzer(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._input_directory_path = self._config_parser.eval(self.__class__.__name__, "input_directory_path")
        self._output_directory_path = self._config_parser.eval(self.__class__.__name__, "output_directory_path")
        self._tar_files_for_downloading_urls = self._config_parser.eval(self.__class__.__name__, "tar_files_for_downloading_urls")
        self._max_num_of_threads = self._config_parser.eval(self.__class__.__name__, "max_num_of_threads")


    def download_graphs_from_website_single_thread(self):
        print("Downloading graphs as tar:")
        for url in self._tar_files_for_downloading_urls:
            html, network_name = self._read_html_file(url)
            li_items = self._get_li_items_from_html(html)

            total_graphs = len(li_items)
            i = 0
            for li_item in li_items:
                i += 1
                msg = "\r Network: {0}, Number of downloaded graphs {1}/{2}".format(network_name, i, total_graphs)
                print(msg, end="")
                tar_file_name = li_item.text

                # downloading the tar and save it
                url_for_downloading = url + "/" + tar_file_name
                url_for_saving = self._input_directory_path + "/" + tar_file_name

                urllib.request.urlretrieve(url_for_downloading, url_for_saving)

    def download_graphs_from_website_parallel(self):
        for url in self._tar_files_for_downloading_urls:
            html, network_name = self._read_html_file(url)
            li_items = self._get_li_items_from_html(html)

            total_graphs = len(li_items)
            i = 0
            for li_item in li_items:
                i += 1
                thread = threading.Thread(target=self._download_graph, args=(url, li_item, network_name, i, total_graphs,))
                thread.start()
                if threading.active_count() == self._max_num_of_threads:  # set maximum threads.
                    thread.join()

    def _download_graph(self, url, li_item, network_name, i, total_graphs):
        msg = "\r Network: {0}, Number of downloaded graphs {1}/{2}".format(network_name, i, total_graphs)
        print(msg, end="")
        tar_file_name = li_item.text

        # downloading the tar and save it
        url_for_downloading = url + "/" + tar_file_name
        url_for_saving = self._input_directory_path + "/" + tar_file_name

        urllib.request.urlretrieve(url_for_downloading, url_for_saving)


    def extract_tar_files_parallel(self):
        print("Extracting tar gz files:")
        tar_gz_files = os.listdir(self._input_directory_path)

        for tar_gz_file in tar_gz_files:
            thread = threading.Thread(target=self._extract_tar_gz_file, args=(tar_gz_file,))
            thread.start()
            if threading.active_count() == self._max_num_of_threads:  # set maximum threads.
                thread.join()

    def extract_tar_files_single_thread(self):
        print("Extract_tar_files_single_thread:")
        tar_gz_files = os.listdir(self._input_directory_path)
        num_of_tar_gz_files = len(tar_gz_files)

        i = 0
        for tar_gz_file in tar_gz_files:
            i += 1

            msg = "\r Extracting file name: {0}/{1}".format(i, num_of_tar_gz_files)
            print(msg, end="")

            self._extract_tar_gz_file(tar_gz_file)

    def _extract_tar_gz_file(self, tar_gz_file):
        #for tar_gz_file in tar_gz_files:
        try:
            if "tar.gz" in tar_gz_file:
                category, sub_category = self._extract_category_and_sub_category_from_tar_gz_file(tar_gz_file)

                new_folder_name = category + "__" + sub_category

                # create new folder in the output directory
                destination_full_path = self._output_directory_path + new_folder_name
                if not os.path.exists(destination_full_path):
                    os.makedirs(destination_full_path)

                # read the file from input folder
                tar = tarfile.open(self._input_directory_path + "/" + tar_gz_file)
                # extract the file in the output folder
                tar.extractall(path=destination_full_path)
                tar.close()
        except:
            logging.error("Extracting problem for file name::{0}".format(tar_gz_file))


    def get_nodes_and_edges(self):
        directory_names = os.listdir(self._input_directory_path)
        graph_tuples = []
        for directory_name in directory_names:
            file_names = os.listdir(self._input_directory_path + directory_name)
            for file_name in file_names:
                if ".sgraph" in file_name:
                    print("File name is: {0}".format(file_name))
                    pattern = "^([^\.]+)__([^\.]+).[^\.]+.([^\.]+).sgraph$"
                    match = re.match(pattern, file_name)
                    group_tuple = match.groups()
                    category = group_tuple[0]
                    sub_category = group_tuple[1]
                    timestamp = group_tuple[2]

                    sub_graph = gl.load_sgraph(self._input_directory_path + directory_name + "/" + file_name)
                    sub_graph.save(self._output_directory_path + file_name + ".csv", format='csv')

                    summary_dict = sub_graph.summary()

                    num_vertices = summary_dict['num_vertices']
                    num_edges = summary_dict['num_edges']

                    tuple = (category, sub_category, timestamp, num_vertices, num_edges)
                    graph_tuples.append(tuple)

        df = pd.DataFrame(graph_tuples, columns=['category', 'sub_category', 'date', 'nodes', 'edges'])
        df.to_csv(self._output_directory_path + "graph_summary.csv")

    def _extract_category_and_sub_category_from_tar_gz_file(self, tar_gz_file):
        category = "Unknown"
        sub_category = "Unknown"

        pattern = "^([^\.]+)__([^\.]+).tar.gz$"
        match = re.match(pattern, tar_gz_file)
        if match is not None:
            group_tuple = match.groups()
            category = group_tuple[0]
            sub_category = group_tuple[1]
        else:
            pattern = "^__([^\.]+).tar.gz$"
            match = re.match(pattern, tar_gz_file)
            if match is not None:
                group_tuple = match.groups()
                sub_category = group_tuple[0]
        return category, sub_category

    def _read_html_file(self, url):
        pattern = "https://dynamics.cs.washington.edu/[^\.]+/([^\.]+)/$"
        match = re.match(pattern, url)
        group_tuple = match.groups()
        network_name = group_tuple[0]

        # Downloading the
        file = urllib.request.urlopen(url)
        html = file.read()
        return html, network_name

    def _get_li_items_from_html(self, html):
        beautiful_soup = BeautifulSoup(html)
        ul = beautiful_soup.find('ul')
        li_items = ul.findAll('li')
        return li_items

