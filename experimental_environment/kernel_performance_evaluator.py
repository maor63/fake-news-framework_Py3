
from preprocessing_tools.abstract_controller import AbstractController
from datetime import datetime
from data_exporter.arff_writer import ArffWriter
from DB.schema_definition import AuthorFeatures
from itertools import combinations
import networkx as nx


class Kernel_Performance_Evaluator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._kernels = self._config_parser.eval(self.__class__.__name__, "kernels")
        self._index_field_for_predictions = self._config_parser.get(self.__class__.__name__, "index_field_for_predictions")
        self._path = self._config_parser.get(self.__class__.__name__, "path")


    def set_up(self):
        pass

    def execute(self, window_start=None):
        labeled_author_dict, unlabeled_author_dict, unlabeled_author_guid_index_field_dict = self._db.create_author_dictionaries(self._index_field_for_predictions, self._domain)
        for kernel in self._kernels:
            current = 0
            start_time = datetime.now()
            authors_features = []
            author_connections = self._db.get_author_connections_by_connection_type(kernel)
            graph = nx.Graph()
            graph.add_weighted_edges_from(author_connections)
            total = graph.number_of_edges()
            print('Start processing '+kernel+'\n ')
            for author_a, author_b, data in graph.edges(data=True):
                if author_a in labeled_author_dict and author_b in labeled_author_dict:
                    if labeled_author_dict[author_a] == labeled_author_dict[author_b]:
                        same_class = 1
                    else:
                        same_class = 0
                    authors_features.append((kernel+'#'+author_a+'#'+author_b, data['weight'], same_class))
                current += 1
                print('\r pairs done ' + str(current) + ' out of ' + str(total) + ' ', end="")
            print('Finished processing '+kernel+' \n ')
            self.save_predictions_to_arff(authors_features, kernel+'_predictions.arff')
            end_time = datetime.now()
            duration = end_time - start_time
            self.save_benchmarks_to_csv((start_time, end_time, duration), kernel + '_benchmarks.csv')

    def save_benchmarks_to_csv(self, data, filename):
        header = 'Start Date, End Date, Duration \n'
        with open(filename, "a") as text_file:
            text_file.write(header)
            line = data[0].strftime("%Y-%m-%d %H:%M:%S") + ',' + data[1].strftime("%Y-%m-%d %H:%M:%S") + ',' + str(data[2])
            text_file.write(line)

    def save_predictions_to_arff(self, data, filename):
        header = ' @RELATION kernel \n '
        header += ' @ATTRIBUTE kernel_value numeric \n '
        header += ' @ATTRIBUTE same_class numeric \n '
        header += ' @DATA \n '

        with open(self._path + filename, "w") as text_file:
            lines = []
            text_file.write(header)
            for row in data:
                line = str(row[1]) + ',' + str(row[2]) + ' \n '
                lines.append(line)
            text_file.writelines(lines)