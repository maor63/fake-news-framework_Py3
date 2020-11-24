
from configuration.config_class import getConfig
import logging
from DB.schema_definition import DB
from logging import config
from commons.commons import *
import os
import csv
import re


class TimelineOverlapVisualizationGenerator():

    def __init__(self):
        config_parser = getConfig()
        logging.config.fileConfig(getConfig().get("DEFAULT", "logger_conf_file"))
        self._db = DB()
        self._db.setUp()
        self._acquired_bad_authors = []
        self._suspected_authors = []
        self.common_posts_threshold = config_parser.eval(self.__class__.__name__, "common_posts_threshold")
        self.output_path = config_parser.eval(self.__class__.__name__, "output_path")
        self.output_dir = config_parser.eval(self.__class__.__name__, "output_dir")
        # self.unlabeled_prediction_threshold = config_parser.eval(self.__class__.__name__, "unlabeled_prediction_threshold")

        if not os.path.exists(self.output_path + "/" + self.output_dir):
            os.makedirs(self.output_path + "/" + self.output_dir)
        self._source_author_destination_author_num_of_mutual_posts_dict = {}


    def setUp(self):
        # csv = open('Labeling_good-bad_unlabeled_author_random_forest_all_features.csv','r')
        # unlabeled_bad_authors = csv.read()
        # unlabeled_bad_authors = [row.split(',') for row in unlabeled_bad_authors.split('\n')]
        # unlabeled_bad_authors = unlabeled_bad_authors[1:len(unlabeled_bad_authors)-1]
        #
        # unlabeled_bad_authors_over_threshold = []
        # for row in unlabeled_bad_authors:
        #     if float(row[5]) >= self.unlabeled_prediction_threshold:
        #         unlabeled_bad_authors_over_threshold.append(row)
        #         self.unlabeled_bad_authors.append(row[1])

        self._acquired_bad_authors = self._db.get_all_acquired_crowdturfer_authors()
        #manually_labeled_bad_actors = self._db.get_all_manually_labeled_bad_actors()
        unlabeled_authors = self._db.get_all_unlabeled_authors()
        self._suspected_authors = unlabeled_authors


    def generate_timeline_overlap_csv(self):
        '''
         Generates 2 csv files representing the overlapping of:
         * labeled-unlabeled bad authors - manually labeled-automatilcally labeled bad actors' timeline overlapping
         * unlabaled-unlabeled authors - automatocally labeled-automatically labeled bad actors' timeline overlapping
         The csv construct an undirected graph of the form <source node>, <target node>, <edge weight>
        '''
        logging.info("Generating timline overlap csv...")
        acquired_to_acquired_overlap_edges = []
        acquired_to_suspected_overlap_edges = []

        authors_count = 1
        self._source_author_destination_author_num_of_mutual_posts_dict = self._create_source_author_destination_author_num_of_mutual_posts_dict()
        logging.info("Analyzing authors overlaps ...")
        potential_bad_actors = []
        for source in self._source_author_destination_author_num_of_mutual_posts_dict:
            logging.info("Analyzing author {0}/{1}".format(authors_count, len(self._source_author_destination_author_num_of_mutual_posts_dict)))
            authors_count += 1
            if source not in self._acquired_bad_authors: #irrelevant author
                continue

            potential_bad_actors = potential_bad_actors + self._prepare_lists_for_csv_export_and_list_of_potential_bad_actors(acquired_to_acquired_overlap_edges, acquired_to_suspected_overlap_edges, source)

           
        potential_bad_actors = list(set(potential_bad_actors))
        print("Num of potential bad actors is: " + str(len(potential_bad_actors)))
        if len(potential_bad_actors) > 0:
            #self._db.update_bad_actor_from_timeline_overlaping(potential_bad_actors)

            self.extract_csv(acquired_to_acquired_overlap_edges, "acquired_to_acquired_overlap.csv")
            self.extract_csv(acquired_to_suspected_overlap_edges, "acquired_to_suspected_overlap.csv")



    def _create_source_author_destination_author_num_of_mutual_posts_dict(self):
        '''
        returns a mapping of source_author_name->target_author_name->#common_posts
        '''
        maximum_overlap = -1

        counter = 1
        #posts_content_screen_name_tuples = self._db._get_posts_content_screen_name_tuples()
        posts_content_screen_name_tuples = []
        cursor = self._db._get_posts_content_screen_name_tuples()
        posts_content_screen_name_tuples_generator = self._db.result_iter(cursor)

        for tuple in posts_content_screen_name_tuples_generator:
            posts_content_screen_name_tuples.append(tuple)

        #logging.info("Analyzing {0} posts".format(len(posts_content_screen_name_tuples)))

        post_content_authors_dictionary = self._create_post_content_authors_dictinary(posts_content_screen_name_tuples)

        for post_content in post_content_authors_dictionary:
            msg1 = "Analyzing post {0}/{1}".format(counter, len(post_content_authors_dictionary))
            print(msg1)
            counter += 1

            authors_names = post_content_authors_dictionary[post_content]
            i = 1
            for source_author in authors_names:
                msg = "Analyzing source {0}/{1} : {2}".format(i, len(authors_names),post_content)
                print(msg)
                i += 1
                for target_author in authors_names:
                    minimal_num_of_mutual_posts = min(authors_names[source_author],authors_names[target_author])
                    self._source_target_author_mapping(source_author,target_author,minimal_num_of_mutual_posts)




        return self._source_author_destination_author_num_of_mutual_posts_dict


    def extract_csv(self, edges, file_name):
        with open(self.output_path + "/" + self.output_dir + "/" + file_name, 'w', newline='') as csvfile:
            fieldnames = ["source", "target", "weight", "type"]
            writer = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for edge in edges:
                writer.writerow(edge + ["Undirected"])

    def _create_post_content_authors_dictinary(self, post_content_and_author_tuple):
        post_content_authors_dictinary = {}
        for post_content,author in post_content_and_author_tuple:
            if post_content is None or len(post_content) < 10: #too short post
                continue
            content = re.sub(r'https?:.*','', post_content, flags=re.MULTILINE)
            if len(content) > 10:
                content = str(content.encode('utf-8'))
                if content not in post_content_authors_dictinary:
                    post_content_authors_dictinary[content] = {}
                if author not in post_content_authors_dictinary[content]:
                    post_content_authors_dictinary[content][author] = 1
                else:
                    post_content_authors_dictinary[content][author] += 1

        return post_content_authors_dictinary


    def _source_target_author_mapping(self, source_author, target_author, minimal_num_of_mutual_posts):
        if source_author >= target_author: # avoid duplicated edges + self edges
            pass
        else:
            if source_author not in self._source_author_destination_author_num_of_mutual_posts_dict:
                self._source_author_destination_author_num_of_mutual_posts_dict[source_author] = {}
            if target_author not in self._source_author_destination_author_num_of_mutual_posts_dict[source_author]:
                self._source_author_destination_author_num_of_mutual_posts_dict[source_author][target_author] = 0
            self._source_author_destination_author_num_of_mutual_posts_dict[source_author][target_author] += minimal_num_of_mutual_posts

            #create the oppesite edge
            if target_author not in self._source_author_destination_author_num_of_mutual_posts_dict:
                self._source_author_destination_author_num_of_mutual_posts_dict[target_author] = {}
            if source_author not in self._source_author_destination_author_num_of_mutual_posts_dict[target_author]:
                self._source_author_destination_author_num_of_mutual_posts_dict[target_author][source_author] = 0
            self._source_author_destination_author_num_of_mutual_posts_dict[target_author][source_author] += minimal_num_of_mutual_posts

    def _prepare_lists_for_csv_export_and_list_of_potential_bad_actors(self, acquired_to_acquired_overlap_edges, acquired_to_suspected_overlap_edges, source):
        potential_bad_actors = []
        for target in self._source_author_destination_author_num_of_mutual_posts_dict[source]:

            overlap_score = self._source_author_destination_author_num_of_mutual_posts_dict[source][target]
            if overlap_score < self.common_posts_threshold:
                continue

            # unlabeled_to_unlabeled_overlap report
            if target in self._acquired_bad_authors:
                acquired_to_acquired_overlap_edges.append([source, target, overlap_score])

            # unlabled_to_labeled_overlap report
            elif target in self._suspected_authors:
                acquired_to_suspected_overlap_edges.append([source, target, overlap_score])
                potential_bad_actors.append(target)

        return potential_bad_actors
