

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from commons.method_executor import Method_Executor
import pandas as pd

from dataset_builder.bot_or_not.botometer_obj import BotometerObj

__author__ = "Aviad Elyashar"

class BotometerEvaluator(Method_Executor):

    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._targeted_class_dict = self._config_parser.eval(self.__class__.__name__, "targeted_class_dict")
        self._divide_lableled_by_percent_training_size = self._config_parser.eval(self.__class__.__name__, "divide_lableled_by_percent_training_size")
        self._num_of_iterations = self._config_parser.eval(self.__class__.__name__, "num_of_iterations")
        self._targeted_class_field_name = self._config_parser.eval(self.__class__.__name__, "targeted_class_field_name")
        self._index_field_name = self._config_parser.eval(self.__class__.__name__, "index_field_name")
        self._path = self._config_parser.eval(self.__class__.__name__, "path")
        self._results_table_file_name = self._config_parser.eval(self.__class__.__name__, "results_table_file_name")

        # in order to reduce thew number of requests to botometer - if you checked once someone keep the result
        self._screen_name_botometer_score_dict = {}

    def divide_to_training_and_test_by_percent_random(self):
        print("Running divide_to_training_and_test_by_percent_random")

        botometer_obj = BotometerObj()
        botometer = botometer_obj.get_botometer()

        authors = self._db.get_authors(self._domain)
        all_authors_df = self._create_authors_df(authors)

        targeted_classes = list(self._targeted_class_dict.keys())
        labeled_authors_df = self._create_labeled_dfs(all_authors_df, targeted_classes)

        self._result_tuples = []

        total_suspended_users = []
        for training_size_percent in self._divide_lableled_by_percent_training_size:
            for i in range(self._num_of_iterations):

                msg = "\r Classifier: {0}, training size: {1}, iteration: {2}".format("BotOrNot", training_size_percent, i)
                print(msg, end="")

                training_set_num_records, original_test_set_num_records, test_df = self._divide_to_training_and_test_sets(training_size_percent, labeled_authors_df, all_authors_df)

                test_screen_names_to_check, all_test_screen_names = self._calculate_screen_names_to_check(test_df)

                self._check_accounts_via_botometer(botometer, test_screen_names_to_check)

                botomoter_results_df = self._set_botomoter_results(all_test_screen_names)

                botomoter_results_df, test_df, suspended_users = self._remove_suspended_users(botomoter_results_df, test_df)
                total_suspended_users += suspended_users
                total_suspended_users = list(set(total_suspended_users))
                actual_test_set_num_records = test_df.shape[0]

                botometer_predictions = botomoter_results_df['botometer_prediction']
                test_ground_truth_series = test_df[self._targeted_class_field_name]

                self._calculate_performance(test_ground_truth_series, botometer_predictions, training_size_percent, training_set_num_records,
                                            original_test_set_num_records, actual_test_set_num_records, i)


        self._print_results(total_suspended_users)

    def predict_with_botometer_on_all_authors(self):
        botometer_obj = BotometerObj()
        botometer = botometer_obj.get_botometer()

        authors = self._db.get_authors_by_domain(self._domain)
        author_screen_names = [author.author_screen_name for author in authors]
        self._check_accounts_via_botometer(botometer, author_screen_names)

        prediction_tuples = []
        for author_screen_name, botometer_score in list(self._screen_name_botometer_score_dict.items()):
            author_type = "good_actor"
            if botometer_score > 0.5:
                author_type = "bad_actor"

            prediction_tuple = (author_screen_name, author_type, botometer_score)
            prediction_tuples.append(prediction_tuple)

        botometer_predictions_df = pd.DataFrame(prediction_tuples, columns=['AccountPropertiesFeatureGenerator_author_screen_name',
                                                                            'predicted', 'prediction'])

        table_name = "unlabeled_predictions"
        botometer_predictions_df.to_csv(self._path + table_name + ".csv", index=False)

        self._db.drop_unlabeled_predictions(table_name)

        engine = self._db.engine
        botometer_predictions_df.to_sql(name=table_name, con=engine)



    def _create_labeled_dfs(self, labeled_authors_df, optional_classes):
        targeted_class_dfs = []
        for optional_class in optional_classes:
            target_class_labeled_authors_df = labeled_authors_df.loc[labeled_authors_df['author_type'] == optional_class]
            targeted_class_dfs.append(target_class_labeled_authors_df)
        return targeted_class_dfs

    def _build_training_set(self, training_size_percent, targeted_class_dfs):
        sample_targeted_class_dfs = []
        for targeted_class_df in targeted_class_dfs:
            #Choosing randonly samples from each class
            sample_targeted_class_df = targeted_class_df.sample(frac=training_size_percent)
            sample_targeted_class_dfs.append(sample_targeted_class_df)
        training_set_df = pd.concat(sample_targeted_class_dfs)
        return training_set_df

    def _replace_nominal_class_to_numeric(self, df):
        for targeted_class, num in self._targeted_class_dict.items():
            df = df.replace(to_replace=targeted_class, value=num)
        return df

    def _remove_suspended_users(self, botometer_df, test_df):
        suspended_users_df = botometer_df[botometer_df['botometer_prediction'] == 2]
        suspended_users = list(suspended_users_df['screen_name'])
        indexes_to_remove = botometer_df[botometer_df['botometer_prediction'] == 2].index.tolist()
        botometer_df = botometer_df.drop(botometer_df.index[indexes_to_remove])
        botometer_df = botometer_df.reset_index()

        test_df = test_df.drop(test_df.index[indexes_to_remove])
        test_df = test_df.reset_index()

        return botometer_df, test_df, suspended_users

    def _create_authors_df(self, authors):
        author_screen_name_author_type_tuples = []
        for author in authors:
            author_screen_name = author.author_screen_name
            author_type = author.author_type

            author_screen_name_author_type_tuple = (author_screen_name, author_type)
            author_screen_name_author_type_tuples.append(author_screen_name_author_type_tuple)

        all_authors_df = pd.DataFrame(author_screen_name_author_type_tuples,
                                      columns=[self._index_field_name, self._targeted_class_field_name])
        return all_authors_df

    def _divide_to_training_and_test_sets(self, training_size_percent, labeled_authors_df, all_authors_df):
        training_df = self._build_training_set(training_size_percent, labeled_authors_df)
        training_set_num_records = training_df.shape[0]

        training_df_indexes = training_df.index.tolist()

        test_df = all_authors_df[~all_authors_df.index.isin(training_df_indexes)]
        original_test_set_num_records = test_df.shape[0]

        test_df = self._replace_nominal_class_to_numeric(test_df)
        return training_set_num_records, original_test_set_num_records, test_df

    def _calculate_screen_names_to_check(self, test_df):
        test_screen_name_series = test_df[self._index_field_name]
        all_test_screen_names = list(test_screen_name_series)
        already_checked_screen_names = list(self._screen_name_botometer_score_dict.keys())
        test_screen_names_to_check = list(set(all_test_screen_names) - set(already_checked_screen_names))
        return test_screen_names_to_check, all_test_screen_names

    def _check_accounts_via_botometer(self, botometer, test_screen_names_to_check):
        # test_screen_names = ['@' + screen_name for screen_name in test_screen_names]
        j = 0
        for screen_name, result_dict in botometer.check_accounts_in(test_screen_names_to_check):
            j += 1
            if 'error' in result_dict:
                bot_score = 2
            else:
                bot_score = result_dict['scores']['universal']

            msg = "\r{0}/{1} screen_name: {2}, botometer_score: {3}, ".format(j, len(test_screen_names_to_check),
                                                                              screen_name, bot_score, )
            print(msg, end="")

            # update the dict in order not to ask again the user for saving time and performance
            self._screen_name_botometer_score_dict[screen_name] = bot_score

    def _set_botomoter_results(self, all_test_screen_names):
        screen_name_botometer_prediction_tuples = []
        # building the series for prediction
        for test_screen_name in all_test_screen_names:
            is_bot = 0
            bot_score = self._screen_name_botometer_score_dict[test_screen_name]
            if bot_score > 0.5 and bot_score < 2:
                is_bot = 1
            elif bot_score == 2:
                is_bot = 2
            screen_name_botometer_prediction_tuple = (test_screen_name, is_bot)
            screen_name_botometer_prediction_tuples.append(screen_name_botometer_prediction_tuple)

        botomoter_results_df = pd.DataFrame(screen_name_botometer_prediction_tuples,
                                            columns=['screen_name', 'botometer_prediction'])
        return botomoter_results_df

    def _calculate_performance(self, test_ground_truth_series, botometer_predictions, training_size_percent, training_set_num_records,
                               original_test_set_num_records, actual_test_set_num_records, i):
        try:
            auc_score = roc_auc_score(test_ground_truth_series, botometer_predictions)
        except:
            auc_score = -1
        accuracy = accuracy_score(test_ground_truth_series, botometer_predictions)
        f1 = f1_score(test_ground_truth_series, botometer_predictions)
        precision = precision_score(test_ground_truth_series, botometer_predictions)
        recall = recall_score(test_ground_truth_series, botometer_predictions)
        conf_matrix = confusion_matrix(test_ground_truth_series, botometer_predictions)

        result_tuple = (
        "Botometer Classifier", training_size_percent, training_set_num_records, original_test_set_num_records,
        actual_test_set_num_records, i, auc_score, accuracy, f1, precision, recall, conf_matrix)

        self._result_tuples.append(result_tuple)

    def _print_results(self, total_suspended_users):
        df = pd.DataFrame(self._result_tuples, columns=['Classifier', "%Training Size", "#Training Set Records",
                                                        "#Original Test Set Records", "#Actual Test Set Records",
                                                        "#Iteration", "AUC", "Accuracy", "F1", "Precision", "Recall",
                                                        "Confusion Matrix"])
        df.to_csv(self._path + self._results_table_file_name, index=None)

        suspended_users_tuples = tuple(total_suspended_users)
        suspended_users_df = pd.DataFrame(suspended_users_tuples, columns=['suspended_screen_name'])
        suspended_users_df.to_csv(self._path + "suspended_users.csv", index=None)
