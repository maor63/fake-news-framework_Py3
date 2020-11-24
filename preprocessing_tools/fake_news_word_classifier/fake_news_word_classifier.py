

import csv
from collections import Counter, defaultdict

import sys
# from scipy.misc import imread
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from commons.commons import convert_claim_to_post
from preprocessing_tools.fake_news_word_classifier.fake_news_base import FakeNewsBase


class FakeNewsClassifier(FakeNewsBase):

    def __init__(self, db):
        super(FakeNewsClassifier, self).__init__(db)
        self._threshold_value = self._config_parser.eval(self.__class__.__name__, "threshold_value")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._false_value = 1
        self._true_value = 0

    def execute(self, window_start=None):
        for action_name in self._feature_list:
            try:
                getattr(self, action_name)()
            except AttributeError as e:
                print('\nError: {0}\n'.format(e.message), file=sys.stderr)

    def classify_by_dictionary(self):
        for targeted_fields_dict in self._targeted_fields:
            print("Get sourse id target dict")
            source_id_target_elements_dict = self._get_source_id_target_elements(targeted_fields_dict)
            source_ids = list(source_id_target_elements_dict.keys())
            source_id_source_element_dict = self._get_source_id_source_element_dict(source_ids, targeted_fields_dict)
            print("Get post dictionary")

            claim_id_claim_dict = {claim.claim_id: claim for claim in self._db.get_claims()}

            claim_to_total_comments = defaultdict()
            claim_to_comment_frec = defaultdict()
            claim_to_comment_counter = defaultdict()
            i = 1
            for source_targets_dict_item in source_id_target_elements_dict.items():
                msg = "\r classify source by target items {0}/{1}".format(i, len(source_id_target_elements_dict))
                print(msg, end='')
                i += 1
                if len(source_targets_dict_item[1]) >= 20:
                    source_id = source_targets_dict_item[0]
                    author = source_id_source_element_dict[source_id]
                    kwargs = self._get_feature_kwargs(source_targets_dict_item, author, targeted_fields_dict)
                    claim_to_comment_frec[source_id] = self._calc_source_id_word_fraction_dict(kwargs)
                    claim_to_comment_counter[source_id] = self._calc_source_id_word_count_dict(kwargs)
                    claim_to_total_comments[source_id] = len(source_targets_dict_item[1])

            print()
            self.predict_results(claim_to_comment_frec, claim_id_claim_dict)
            self._write_results_to_csv(claim_to_total_comments, claim_to_comment_frec, claim_to_comment_counter,
                                       claim_id_claim_dict)

    def predict_results(self, claim_to_comment_frec, claim_id_claim_dict):
        predicted = self._get_predictions(claim_to_comment_frec, claim_id_claim_dict)
        ground_truth = self._get_ground_truth(claim_to_comment_frec, claim_id_claim_dict)
        FN, FP, count, total = self._get_results_for_predictions(claim_to_comment_frec, ground_truth, predicted)

        print("Pred acc: {0}".format(str(count / total)))
        print("FP acc: {0}".format(str(FP)))
        print("FN acc: {0}".format(str(FN)))
        print("total: {0}".format(str(total)))
        print("total claims: {0}".format(str(len(ground_truth))))
        accuracy = accuracy_score(ground_truth, predicted)
        print("Accuracy: {0}".format(str(accuracy)))
        roc = roc_auc_score(ground_truth, predicted)
        print("Roc Auc score: {0}".format(str(roc)))

        precision = precision_score(ground_truth, predicted)
        print("Precision: {0}".format(str(precision)))

        recall = recall_score(ground_truth, predicted)
        print("Recall: {0}".format(str(recall)))

        columns = ["fraction sum thresh", "accuracy", "AUC", "Precision", "Recall", "FP (think bad but good)", "FN (think good but bad)",
                   "total claims"]
        f = open(self._output_path + "/fake_news_classifier_results.csv", 'wb')
        csv_normal = csv.writer(f, delimiter=",")
        csv_normal.writerow(columns)
        row = [self._threshold_value, accuracy, roc, precision, recall, FP, FN, len(ground_truth)]
        csv_normal.writerow(row)
        f.close()

    def _get_results_for_predictions(self, claim_to_comment_frec, ground_truth, predicted):
        posts_ids = list(claim_to_comment_frec.keys())
        count = 0
        total = 0.0
        FP = 0
        FN = 0
        for i, x in enumerate(predicted):
            if x == self._false_value:
                total += 1
                if ground_truth[i] == self._false_value:
                    count += 1
                elif ground_truth[i] == self._true_value:
                    print(posts_ids[i])
                    FP += 1
            elif x == self._true_value:
                if ground_truth[i] == self._false_value:
                    FN += 1
        return FN, FP, count, total

    def _get_ground_truth(self, claim_to_comment_frec, claim_id_claim_dict):
        ground_truth = []
        for claim_id in claim_to_comment_frec:
            claim_verdict = claim_id_claim_dict[claim_id].verdict
            if claim_verdict in self._bad:
                ground_truth.append(self._false_value)
            elif claim_verdict in self._good:
                ground_truth.append(self._true_value)
        return ground_truth

    def _get_predictions(self, claim_to_comment_frec, claim_id_claim_dict):
        predicted = []
        for claim_id in claim_to_comment_frec:
            claim_verdict = claim_id_claim_dict[claim_id].verdict
            if claim_verdict in self._bad or claim_verdict in self._good:
                val_sum = sum(claim_to_comment_frec[claim_id].values())
                if val_sum > self._threshold_value:  # [0.02, 0.016] give best results 0.018 best
                    predicted.append(self._false_value)
                else:
                    predicted.append(self._true_value)
        return predicted

    def _write_results_to_csv(self, claim_to_total_comments, claim_to_comment_frec, claim_to_comment_counter,
                              claim_id_claim_dict):
        columns = ['claim_id', 'content', 'type', 'total_comments'] + list(self._fake_news_dictionary) + ['sum']
        f_normal = open(self._output_path + "/output_count.csv", "wb")
        f_percent = open(self._output_path + "/output_percentage.csv", "wb")
        csv_normal = csv.writer(f_normal, delimiter=",")
        csv_percent = csv.writer(f_percent, delimiter=",")
        csv_normal.writerow(columns)
        csv_percent.writerow(columns)
        normal_rows = []
        precent_rows = []
        for claim_id in claim_to_comment_frec:
            row = [claim_id, claim_id_claim_dict[claim_id].description, claim_id_claim_dict[claim_id].verdict,
                   claim_to_total_comments[claim_id]]
            word_count = [claim_to_comment_counter[claim_id][x] for x in self._fake_news_dictionary]
            row_numbers = row + word_count + [sum(word_count)]
            word_frec = [float(claim_to_comment_frec[claim_id][x]) for x in self._fake_news_dictionary]
            row_percent = row + word_frec + [sum(word_frec)]
            normal_rows.append(row_numbers)
            precent_rows.append(row_percent)
        csv_normal.writerows(normal_rows)
        csv_percent.writerows(precent_rows)
        f_normal.close()
        f_percent.close()
