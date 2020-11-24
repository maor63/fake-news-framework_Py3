
from commons.method_executor import Method_Executor
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd

class FakeAndRealNewsPromoterLabelAssigner(Method_Executor):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._actions = self._config_parser.eval(self.__class__.__name__, "actions")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._fake_news_labels = self._config_parser.eval(self.__class__.__name__, "fake_news_labels")
        self._real_news_labels = self._config_parser.eval(self.__class__.__name__, "real_news_labels")
        self._min_num_of_distinct_claims_author_involved = self._config_parser.eval(self.__class__.__name__, "min_num_of_distinct_claims_author_involved")
        self._min_num_of_published_posts = self._config_parser.eval(self.__class__.__name__, "min_num_of_published_posts")
        self._threshold = self._config_parser.eval(self.__class__.__name__, "threshold")

        self._original_fake_news_label = "fake_news"
        self._original_real_news_label = "real_news"

        self._fake_news_promoter_label = "fake_news_promoter"
        self._real_news_promoter_label = "real_news_promoter"



    def setUp(self):
        self._verdicts_columns = self._fake_news_labels + self._real_news_labels
        self._author_guid_verdicts_columns = ['author_guid'] + self._verdicts_columns + ['num_of_distinct_claims',
                                                                                         'num_of_posts',
                                                                                         'author_sub_type']

    def _organize_author_guid_by_verdicts_by_statistics(self, author_guid, claim_by_author_guid_df):
        author_guid_by_verdicts = []
        author_guid_by_verdicts.append(author_guid)
        for i, verdict in enumerate(self._verdicts_columns):
            author_guid_verdict_df = claim_by_author_guid_df[claim_by_author_guid_df['verdict'] == verdict]
            num_of_posts_by_verdict = len(author_guid_verdict_df.index)
            author_guid_by_verdicts.append(num_of_posts_by_verdict)
        author_guid_by_verdicts_tuple = tuple(author_guid_by_verdicts)
        self._author_guid_by_verdicts_dict[author_guid] = author_guid_by_verdicts_tuple

    def _convert_df_to_binary_problem(self, claim_by_author_guid_df):
        for fake_news_label in self._fake_news_labels:
            claim_by_author_guid_df = claim_by_author_guid_df.replace(fake_news_label, self._original_fake_news_label)
        for real_news_label in self._real_news_labels:
            claim_by_author_guid_df = claim_by_author_guid_df.replace(real_news_label, self._original_real_news_label)

        return claim_by_author_guid_df

    def label_fake_and_real_news_promoters_by_conditions(self):
        claim_post_author_connections = self._db.get_claim_post_author_connections_with_verdict()

        self._claim_post_author_connection_df = pd.DataFrame(claim_post_author_connections,
                                                             columns=['claim_id', 'post_id', 'author_guid', 'verdict'])

        authors = self._db.get_authors()

        author_claim_df_results = pd.DataFrame()

        self._author_guid_by_verdicts_dict = {}
        self._author_guid_by_binary_verdict_tuples = []
        author_guid_author_type_tuples = []
        for i, author in enumerate(authors):
            msg = "\r Analyzing author {0}/{1}".format(i, len(authors))
            print(msg, end='')

            author_guid = author.author_guid

            claim_by_author_guid_df = self._claim_post_author_connection_df[
                self._claim_post_author_connection_df['author_guid'] == author_guid]
            claim_by_author_guid_df = claim_by_author_guid_df.reset_index()
            num_of_posts = len(claim_by_author_guid_df.index)

            self._organize_author_guid_by_verdicts_by_statistics(author_guid, claim_by_author_guid_df)

            distinct_claims_df = claim_by_author_guid_df.groupby(
                ['author_guid', 'claim_id', 'verdict']).size().reset_index(name='counts')
            author_claim_df_results = author_claim_df_results.append(distinct_claims_df, ignore_index=True)

            num_of_distinct_claims = len(distinct_claims_df.index)
            if num_of_distinct_claims >= self._min_num_of_distinct_claims_author_involved and \
                            num_of_posts >= self._min_num_of_published_posts:

                claim_by_author_guid_df = self._convert_df_to_binary_problem(claim_by_author_guid_df)

                author_guid_by_verict_df = claim_by_author_guid_df[
                    claim_by_author_guid_df['verdict'] == self._original_fake_news_label]
                author_guid_by_verict_df = author_guid_by_verict_df.reset_index()
                num_of_fake_news_posts = len(author_guid_by_verict_df.index)

                author_guid_by_verict_df = claim_by_author_guid_df[
                    claim_by_author_guid_df['verdict'] == self._original_real_news_label]
                author_guid_by_verict_df = author_guid_by_verict_df.reset_index()
                num_of_real_news_posts = len(author_guid_by_verict_df.index)

                fake_news_posts_distribution = num_of_fake_news_posts / float(num_of_posts)
                real_news_posts_distribution = num_of_real_news_posts / float(num_of_posts)

                if fake_news_posts_distribution >= self._threshold:
                    author.author_type = self._fake_news_promoter_label
                elif real_news_posts_distribution >= self._threshold:
                    author.author_type = self._real_news_promoter_label

                author_guid_binary_verdict = (author_guid, num_of_distinct_claims, num_of_fake_news_posts,
                                              num_of_real_news_posts, author.author_type)
                self._author_guid_by_binary_verdict_tuples.append(author_guid_binary_verdict)

            author_guid_by_verict_tuple = self._author_guid_by_verdicts_dict[author_guid]
            author_guid_by_verict_tuple += (num_of_distinct_claims, num_of_posts, author.author_type,)
            self._author_guid_by_verdicts_dict[author_guid] = author_guid_by_verict_tuple

            author_guid_author_type_tuple = (author_guid, author.author_type)
            author_guid_author_type_tuples.append(author_guid_author_type_tuple)

        self._db.addPosts(authors)

        author_guid_by_verdicts_tuples = list(self._author_guid_by_verdicts_dict.values())

        author_guid_by_verdicts_results_df = pd.DataFrame(author_guid_by_verdicts_tuples,
                                                          columns=self._author_guid_verdicts_columns)
        author_guid_by_verdicts_results_df.to_csv(self._output_path + "author_guid_by_verdicts_results.csv", index=False)

        author_guid_author_type_results_df = pd.DataFrame(author_guid_author_type_tuples,
                                                              columns=['author_guid', 'author_type'])
        author_guid_author_type_results_df.to_csv(self._output_path + "author_guid_author_type_results.csv", index=False)

        author_guid_binary_verdict_df = pd.DataFrame(self._author_guid_by_binary_verdict_tuples,
                                                              columns=['author_guid', 'num_of_distinct_claims',
                                                                       'fake_news_num_of_posts', 'real_news_num_of_posts',
                                                                       'author_type'])
        author_guid_binary_verdict_df.to_csv(self._output_path + "author_guid_binary_verdict_results.csv",
                                                      index=False)


    def get_author_statistics(self):
        claim_post_author_connections = self._db.get_claim_post_author_connections_with_verdict()

        self._claim_post_author_connection_df = pd.DataFrame(claim_post_author_connections,
                                                             columns=['claim_id', 'post_id', 'author_guid', 'verdict'])


        authors = self._db.get_authors()

        author_claim_df_results = pd.DataFrame()

        self._author_guid_by_verdicts_dict = {}
        self._author_guid_by_binary_verdict_tuples = []
        author_guid_author_type_tuples = []
        for i, author in enumerate(authors):
            msg = "\r Analyzing author {0}/{1}".format(i, len(authors))
            print(msg, end='')

            author_guid = author.author_guid

            claim_by_author_guid_df = self._claim_post_author_connection_df[
                self._claim_post_author_connection_df['author_guid'] == author_guid]
            claim_by_author_guid_df = claim_by_author_guid_df.reset_index()
            num_of_posts = len(claim_by_author_guid_df.index)

            self._organize_author_guid_by_verdicts_by_statistics(author_guid, claim_by_author_guid_df)

            distinct_claims_df = claim_by_author_guid_df.groupby(
                ['author_guid', 'claim_id', 'verdict']).size().reset_index(name='counts')
            author_claim_df_results = author_claim_df_results.append(distinct_claims_df, ignore_index=True)

            num_of_distinct_claims = len(distinct_claims_df.index)
            if num_of_distinct_claims >= self._min_num_of_distinct_claims_author_involved and \
                            num_of_posts >= self._min_num_of_published_posts:

                claim_by_author_guid_df = self._convert_df_to_binary_problem(claim_by_author_guid_df)

                author_guid_by_verict_df = claim_by_author_guid_df[
                    claim_by_author_guid_df['verdict'] == self._original_fake_news_label]
                author_guid_by_verict_df = author_guid_by_verict_df.reset_index()
                num_of_fake_news_posts = len(author_guid_by_verict_df.index)

                author_guid_by_verict_df = claim_by_author_guid_df[
                    claim_by_author_guid_df['verdict'] == self._original_real_news_label]
                author_guid_by_verict_df = author_guid_by_verict_df.reset_index()
                num_of_real_news_posts = len(author_guid_by_verict_df.index)

                fake_news_posts_distribution = num_of_fake_news_posts / float(num_of_posts)
                real_news_posts_distribution = num_of_real_news_posts / float(num_of_posts)

                #if fake_news_posts_distribution >= self._threshold:
                #    author.author_type = self._fake_news_promoter_label
                #elif real_news_posts_distribution >= self._threshold:
                #    author.author_type = self._real_news_promoter_label


                df = claim_by_author_guid_df.groupby(['claim_id', 'verdict']).count().reset_index().groupby(
                    ['verdict']).count().reset_index()

                num_of_fake_news_claims = df[(df["verdict"] == 'fake_news')]['claim_id'].values[0]
                num_of_real_news_claims = df[(df["verdict"] == 'real_news')]['claim_id'].values[0]


                author_guid_binary_verdict = (author_guid, num_of_distinct_claims, num_of_fake_news_claims,
                                              num_of_real_news_claims,
                                              num_of_fake_news_posts,
                                              num_of_real_news_posts, author.author_type)
                self._author_guid_by_binary_verdict_tuples.append(author_guid_binary_verdict)

            author_guid_by_verict_tuple = self._author_guid_by_verdicts_dict[author_guid]
            author_guid_by_verict_tuple += (num_of_distinct_claims, num_of_posts, author.author_type,)
            self._author_guid_by_verdicts_dict[author_guid] = author_guid_by_verict_tuple

            author_guid_author_type_tuple = (author_guid, author.author_type)
            author_guid_author_type_tuples.append(author_guid_author_type_tuple)

        #self._db.addPosts(authors)

        author_guid_by_verdicts_tuples = list(self._author_guid_by_verdicts_dict.values())

        author_guid_by_verdicts_results_df = pd.DataFrame(author_guid_by_verdicts_tuples,
                                                          columns=self._author_guid_verdicts_columns)
        author_guid_by_verdicts_results_df.to_csv(self._output_path + "author_guid_by_verdicts_results.csv",
                                                  index=False)

        author_guid_author_type_results_df = pd.DataFrame(author_guid_author_type_tuples,
                                                          columns=['author_guid', 'author_type'])
        author_guid_author_type_results_df.to_csv(self._output_path + "author_guid_author_type_results.csv",
                                                  index=False)

        author_guid_binary_verdict_df = pd.DataFrame(self._author_guid_by_binary_verdict_tuples,
                                                     columns=['author_guid', 'num_of_distinct_claims',
                                                              'num_of_fake_news_claims', 'num_of_real_news_claims',
                                                              'fake_news_num_of_posts', 'real_news_num_of_posts',
                                                              'author_type'])
        author_guid_binary_verdict_df.to_csv(self._output_path + "author_guid_binary_verdict_results.csv",
                                             index=False)


    def delete_author_sub_type(self):
        self._db.delete_author_sub_type()
