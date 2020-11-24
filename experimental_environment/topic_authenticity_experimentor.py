from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as conf_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import random

from commons.commons import calculate_correctly_and_not_correctly_instances
from commons.method_executor import Method_Executor
import pandas as pd
import json

from preprocessing_tools.abstract_controller import AbstractController

__author__ = "Aviad Elyashar"

class TopicAuthenticityExperimentor(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._classification_method = self._config_parser.eval(self.__class__.__name__, "classification_method")
        self._path = self._config_parser.eval(self.__class__.__name__, "path")
        self._good_buckets = self._config_parser.eval(self.__class__.__name__, "good_buckets")
        self._bad_buckets = self._config_parser.eval(self.__class__.__name__, "bad_buckets")
        self._input_topic_statistics_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                          "input_topic_statistics_file_name")
        self._claim_id_topic_num_dict_json_file_name = self._config_parser.eval(self.__class__.__name__,
                                                                                "claim_id_topic_num_dict_json_file_name")
        self._targeted_classes_dict = self._config_parser.eval(self.__class__.__name__,
                                                                                "targeted_classes_dict")


    def execute(self, window_start):
        topic_statistics_df, claim_verdict_df = self._create_dfs_for_calculations()
        ground_truth_claim_verdict = claim_verdict_df['verdict']

        actual_predictions, good_claims_probabilities = self.predict_based_on_given_method(topic_statistics_df,
                                                                                      claim_verdict_df)

        self._calculate_performance(ground_truth_claim_verdict, good_claims_probabilities, actual_predictions)
        self._print_predictions(claim_verdict_df, good_claims_probabilities)

    def _convert_verdict_class_to_nominal(self, claim_verdict_df):
        for targeted_class, num in self._targeted_classes_dict.items():
            claim_verdict_df = claim_verdict_df.replace(to_replace=targeted_class, value=num)
        return claim_verdict_df

    def _remove_claims_not_in_targeted_classes(self, claim_verdict_df):
        targeted_classes = list(self._targeted_classes_dict.keys())
        targeted_classes_set = set(targeted_classes)
        claim_verdict_df = claim_verdict_df[claim_verdict_df.verdict.isin(targeted_classes_set)]
        return claim_verdict_df

    def _calculate_sum_of_posts_or_authors_by_buckets(self, topic_statistics_by_topic, buckets):
        topic_statistics_by_buckets = topic_statistics_by_topic[topic_statistics_by_topic.buckets.isin(buckets)]

        num_posts_or_authors_per_topic_in_given_buckets_series = topic_statistics_by_buckets[self._classification_method]
        sum_of_posts_or_authors = num_posts_or_authors_per_topic_in_given_buckets_series.sum()
        return sum_of_posts_or_authors


    def _calculate_performance(self, ground_truth_claim_verdict, good_claims_probabilities, actual_predictions):
        result_performance_tuples = []

        try:
            auc_score = roc_auc_score(ground_truth_claim_verdict, good_claims_probabilities)
        except:
            auc_score = -1
        accuracy = accuracy_score(ground_truth_claim_verdict, actual_predictions)
        f1 = f1_score(ground_truth_claim_verdict, actual_predictions)
        precision = precision_score(ground_truth_claim_verdict, actual_predictions)
        recall = recall_score(ground_truth_claim_verdict, actual_predictions)
        #lrap = label_ranking_average_precision_score(ground_truth_claim_verdict, good_claims_probabilities)
        #lrl = label_ranking_loss(ground_truth_claim_verdict, good_claims_probabilities)
        confusion_matrix = conf_matrix(ground_truth_claim_verdict, actual_predictions)

        num_correctly, num_incorrectly = calculate_correctly_and_not_correctly_instances(confusion_matrix)

        result_performance_tuple = ("classification_by_{0}".format(self._classification_method), num_correctly,
                                    num_incorrectly, auc_score, accuracy, f1, precision, recall, confusion_matrix)
                                    #accuracy, f1, precision, recall, lrap, lrl, confusion_matrix)


        result_performance_tuples.append(result_performance_tuple)

        performance_result_df = pd.DataFrame(result_performance_tuples, columns=['Method', '#Correctly', '#Incorrectly',
                                                                                 'AUC', 'Accuracy','F1', 'Precision',
                                                                                 'Recall', 'Confusion Matrix'])

        performance_result_df.to_csv(self._path + "classifcation_by_{0}_performance_results.csv".format(self._classification_method))

    def _create_dfs_for_calculations(self):
        topic_statistics_df = pd.read_csv(self._path + self._input_topic_statistics_file_name)

        with open(self._path + self._claim_id_topic_num_dict_json_file_name) as json_file:
            self._claim_id_topic_num_dict = json.load(json_file)

        claim_ids = list(self._claim_id_topic_num_dict.keys())
        claim_ids_set = set(claim_ids)

        all_claims = self._db.get_claims()
        claims = [claim for claim in all_claims if claim.claim_id in claim_ids_set]

        claim_verdict_tuples = []
        for claim in claims:
            claim_id = claim.claim_id
            topic_num = self._claim_id_topic_num_dict[claim_id]
            verdict = claim.verdict
            title = claim.title

            claim_verdict_tuple = (claim_id, topic_num, title, verdict)

            claim_verdict_tuples.append(claim_verdict_tuple)

        claim_verdict_df = pd.DataFrame(claim_verdict_tuples, columns=['claim_id', 'topic_num', 'title', 'verdict'])

        # remove unproven and mixture classes before calculation performance
        claim_verdict_df = self._remove_claims_not_in_targeted_classes(claim_verdict_df)
        claim_verdict_df = self._convert_verdict_class_to_nominal(claim_verdict_df)

        return topic_statistics_df, claim_verdict_df

    def predict_based_on_given_method(self, topic_statistics_df, claim_verdict_df):
        # AUC is calculated by one of the probabilities class. just one of them
        good_claims_probabilities = []
        actual_predictions = []
        for index, row in claim_verdict_df.iterrows():
            topic_num = row['topic_num']
            topic_statistics_by_topic = topic_statistics_df[topic_statistics_df['topic_number'] == topic_num]

            sum_of_good_posts_or_authors = self._calculate_sum_of_posts_or_authors_by_buckets(topic_statistics_by_topic,
                                                                                              self._good_buckets)
            sum_of_bad_posts_or_authors = self._calculate_sum_of_posts_or_authors_by_buckets(topic_statistics_by_topic,
                                                                                             self._bad_buckets)

            total = sum_of_good_posts_or_authors + sum_of_bad_posts_or_authors

            if total != 0:
                good_probability = sum_of_good_posts_or_authors / float(total)
                good_claims_probabilities.append(good_probability)

                if good_probability >= 0.5:
                    actual_predictions.append(0)
                else:
                    actual_predictions.append(1)
            else:
                random_prediction = random.randint(0, 1)
                actual_predictions.append(random_prediction)

                good_claims_probabilities.append(0.5)

        return actual_predictions, good_claims_probabilities

    def _print_predictions(self, claim_verdict_df, good_claims_probabilities):

        actual_predictions = claim_verdict_df.pop('verdict')
        claim_verdict_df['prob_to_be_real_news_promoter'] = good_claims_probabilities

        claim_verdict_df['verdict'] = actual_predictions

        claim_verdict_df.to_csv(self._path +
                                "predictions_for_evaluation_classification_by_{0}.csv".format(self._classification_method))


