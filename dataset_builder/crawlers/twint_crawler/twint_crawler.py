from __future__ import print_function
import numpy as np
import os
from commons.method_executor import Method_Executor
from commons.commons import count_down_time
import pandas as pd
from tqdm import tqdm
import datetime
import twint
from commons.commons import *
from twint.token import RefreshTokenException
import nest_asyncio
nest_asyncio.apply()

__author__ = "Aviad Elyashar"


class TwintCrawler(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        #self._output_path = self._config_parser.eval(self.__class__.__name__, "output_path")
        self._output_file_path = self._config_parser.get(self.__class__.__name__, "output_file_path")
        self._since = self._config_parser.get(self.__class__.__name__, "since")
        self._until = self._config_parser.get(self.__class__.__name__, "until")


    def _save_timeline_by_author_name(self, t, author_name, current_path):
        # t.Search = "from:@amit_segal"
        t.Search = author_name
        # t.Limit = None  # number of Tweets to scrape
        t.Store_csv = True  # store tweets in a csv file
        t.Since = self._since
        t.Until = self._until

        #t.Output = self._output_file_path + f"{author_name[6::]}.csv"  # path to csv file
        t.Output = current_path + f"{author_name[6::]}.csv"  # path to csv file
        twint.run.Search(t)

    def run(self):
        #current_date = str(datetime.datetime.now())[0: 11]
        authors = self._db.get_authors()
        authors_for_twint = [f"from:@{author.author_screen_name}" for author in authors]

        t = twint.Config()

        while True:
            print("current_path: " + "{0}-{1}".format(self._since, self._until))
            current_path = os.path.join(self._output_file_path, "{0}-{1}/".format(self._since, self._until))
            os.mkdir(current_path)

            for author in tqdm(authors_for_twint):
                self._save_timeline_by_author_name(t, author, current_path)

            self.__update_since_and_until_for_new_iteration()

            count_down_time(100)
            print("Done iteration !!!")


            # #t.Search = "from:@amit_segal"
            # t.Search = author
            # #t.Limit = None  # number of Tweets to scrape
            # t.Store_csv = True  # store tweets in a csv file
            # t.Since = self._since
            # t.Until = self._until
            #
            #
            # t.Output = self._output_file_path + f"{author[6::]}.csv"  # path to csv file
            # twint.run.Search(t)

    def fill_timelines_for_missing_users(self):
        authors = self._db.get_authors()
        author_screen_names = [author.author_screen_name for author in authors]

        already_crawled_timeline_files = os.listdir(self._output_file_path)
        already_crawled_users = [file.split(".csv")[0] for file in already_crawled_timeline_files]

        missing_author_screen_names = list(set(author_screen_names) - set(already_crawled_users))


        authors_for_twint = [f"from:@{screen_name}" for screen_name in missing_author_screen_names]

        t = twint.Config()

        for author in tqdm(authors_for_twint):
            try:
                self._save_timeline_by_author_name(t, author)

            except RefreshTokenException:
                print(author)
                count_down_time(900)
                self._save_timeline_by_author_name(t, author)

            except TimeoutError as e:
                print(e)
                print(author)
                count_down_time(900)
                self._save_timeline_by_author_name(t, author)

        print("Done!!!")

    def get_user_oldest_post(self):
        current_date = str(datetime.datetime.now())[0: 11]
        files = os.listdir(self._output_file_path)

        dfs = []
        for file in files:
            full_path = self._output_file_path + file

            df = pd.read_csv(full_path)
            dfs.append(df)

        timelines_df = pd.concat(dfs)
        short_timelines_df = timelines_df[["username", "date"]]
        short_timelines_df = short_timelines_df.rename(columns={'date': 'str_date'})
        short_timelines_df['date'] = pd.to_datetime(short_timelines_df['str_date'])
        #df.agg(Minimum_Date=('Date', np.min), Maximum_Date=('Date', np.max))

        user_min_max_post_date_df = short_timelines_df.groupby(["username"]).agg(Minimum_Date=('date', np.min),
                                                                  Maximum_Date=('date', np.max,))

        user_post_count_df = short_timelines_df.groupby(["username"]).count()

        statistics_df = pd.merge(user_min_max_post_date_df, user_post_count_df, on="username")


        #df.groupby(["username"]).agg({'date': [np.min, np.max]})

        #grouped_df = short_timelines_df.groupby(["username"]).min("date")
        statistics_df.to_csv(self._output_file_path + f"statistics_{current_date}.csv")
        print("Done!!!!")

    def __update_since_and_until_for_new_iteration(self):
        format_date = '%Y-%m-%d'
        until_date = str_to_date(self._until, formate=format_date)
        since_date_updated_date = until_date + timedelta(days=1)
        since_date_updated_str = date_to_str(since_date_updated_date, formate=format_date)

        now = datetime.datetime.now()
        until_date_updated_str = date_to_str(now, formate=format_date)

        self._since = since_date_updated_str
        self._until = until_date_updated_str

