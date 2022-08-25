from __future__ import print_function
import numpy as np
import os
from commons.method_executor import Method_Executor
from commons.commons import count_down_time
import pandas as pd
from tqdm import tqdm
from DB.schema_definition import Term, Term_Tweet_Connection, Post, Author, PostUserMention, Post_citation, Image_Tags
import datetime
import re
import twint
from commons.commons import *
from twint.token import RefreshTokenException
from aiohttp import ClientConnectorError
import nest_asyncio
nest_asyncio.apply()

__author__ = "Aviad Elyashar"


class TwintCrawler(Method_Executor):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._output_file_path = self._config_parser.get(self.__class__.__name__, "output_file_path")
        self._since = self._config_parser.get(self.__class__.__name__, "since")
        self._until = self._config_parser.get(self.__class__.__name__, "until")
        self.__not_timelines_screen_name_file_name = self._config_parser.get(self.__class__.__name__,
                                                                             "not_timelines_screen_name_file_name")

        self.__checked_authors = []
        self.__checked_users_df = pd.DataFrame([], columns=["Username"])

    def get_tweets_by_terms(self):
        t = twint.Config()

        current_path = self._output_file_path + "crawled_tweets\\"
        if not os.path.exists(current_path):
            os.mkdir(current_path)

        print("current_path: " + "{0}-{1}".format(self._since, self._until))
        current_path = os.path.join(current_path, "{0}-{1}\\".format(self._since, self._until))

        if not os.path.exists(current_path):
            os.mkdir(current_path)

        df = pd.read_csv(self._output_file_path + "hashtags_from_Abigail.csv", encoding= 'unicode_escape')
        hashtags = df["hashtag"].tolist()
        hashtags = [hashtag.lower().strip() for hashtag in hashtags]

        for hashtag in tqdm(hashtags):
            try:
                self.__save_tweets_by_term(t, hashtag, current_path)

            except RefreshTokenException:
                print(hashtag)
                count_down_time(900)
                self.__save_tweets_by_term(t, hashtag, current_path)

            except TimeoutError as e:
                print(e)
                print(hashtag)
                count_down_time(900)
                self.__save_tweets_by_term(t, hashtag, current_path)

        print("Done!!!")


    def __save_tweets_by_term(self,t, term, current_path):
        t.Search = term
        t.Store_csv = True  # store tweets in a csv file
        t.Since = self._since
        t.Until = self._until
        t.Output = f"{current_path}{term}.csv"  # path to csv file
        twint.run.Search(t)

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
            if not os.path.exists(current_path):
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

    def __handle_exception(self, t, author, current_path):
        print(author)
        count_down_time(900)

        df = pd.DataFrame(self.__checked_authors, columns=["Username"])
        self.__checked_users_df = self.__checked_users_df.append(df, ignore_index=True)
        self.__checked_users_df.to_csv(self._output_file_path + "checked_author_screen_names.csv", index=False)
        self.__checked_authors = []
        del df

        #self._save_timeline_by_author_name(t, author, current_path)

    def fill_timelines_for_missing_users(self):

        authors = self._db.get_authors()
        author_screen_names = [author.author_screen_name for author in authors]

        current_path = os.path.join(self._output_file_path, "{0}-{1}/".format(self._since, self._until))

        already_crawled_timeline_files = os.listdir(current_path)
        already_crawled_users = [file.split(".csv")[0] for file in already_crawled_timeline_files]

        not_timelines_screen_names = []
        full_path_file = self._output_file_path + self.__not_timelines_screen_name_file_name
        if os.path.isfile(full_path_file):
            df = pd.read_csv(full_path_file)
            not_timelines_screen_names = df["Username"].tolist()


        missing_author_screen_names = list(set(author_screen_names) - set(already_crawled_users) - set(not_timelines_screen_names))


        authors_for_twint = [f"from:@{screen_name}" for screen_name in missing_author_screen_names]

        t = twint.Config()

        while True:
            print("current_path: " + "{0}-{1}".format(self._since, self._until))
            current_path = os.path.join(self._output_file_path, "{0}-{1}/".format(self._since, self._until))
            if not os.path.exists(current_path):
                os.mkdir(current_path)

            for author in tqdm(authors_for_twint):
                try:
                    author_name = author.split("from:@")[1]
                    self.__checked_authors.append(author_name)
                    self._save_timeline_by_author_name(t, author, current_path)

                except RefreshTokenException as e:
                    print(e)
                    self.__handle_exception(t, author, current_path)
                    # print(author)
                    # count_down_time(900)
                    try:
                        self._save_timeline_by_author_name(t, author, current_path)
                    except RefreshTokenException as e:
                        self.__handle_exception(t, author, current_path)
                        self._save_timeline_by_author_name(t, author, current_path)

                except TimeoutError as e:
                    print(e)
                    self.__handle_exception(t, author, current_path)
                    # print(author)
                    # count_down_time(900)
                    self._save_timeline_by_author_name(t, author, current_path)

                except ClientConnectorError as e:
                    print(e)
                    self.__handle_exception(t, author, current_path)
                    #print(e)
                    # print(author)
                    # count_down_time(900)
                    self._save_timeline_by_author_name(t, author, current_path)

                except:
                    self.__handle_exception(t, author, current_path)
                    # print(author)
                    # count_down_time(900)
                    self._save_timeline_by_author_name(t, author, current_path)


            self.__update_since_and_until_for_new_iteration()
            authors_for_twint = [f"from:@{screen_name}" for screen_name in author_screen_names]
            count_down_time(100)

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


    def __create_term(self, hashtag):
        term = Term()

        term.term_id = self.__next_term_id
        self.__next_term_id += 1

        term.description = hashtag
        return term

    def __convert_df_row_to_author(self, row):
        author = Author()

        author.domain = self._domain

        user_id = row["user_id"]
        author.author_osn_id = user_id

        username = row["username"]
        author.author_screen_name = username
        author.name = username

        author.author_guid = compute_author_guid_by_author_name(username)

        name = row["name"]
        author.author_full_name = name


        timezone = row["timezone"]
        author.time_zone = timezone

        place = row["place"]
        author.location = place

        geo = row["geo"]
        author.geo_enabled = geo

        reply_count = row["replies_count"]
        author.replies_count = reply_count

        retweets_count = row["retweets_count"]
        author.retweets_count = retweets_count

        likes_count = row["likes_count"]
        author.likes_count = likes_count

        author.original_tweet_importer_insertion_date = datetime.datetime.now()

        return author

    def __convert_row_to_user_mentions(self, row, post):
        user_mentions = []
        mentions = row["mentions"]
        for mention_dict in mentions:
            post_user_mention = PostUserMention()

            post_user_mention.post_guid = post.post_guid
            post_user_mention.user_mention_twitter_id = post.post_id

            screen_name = mention_dict["screen_name"]
            post_user_mention.user_mention_screen_name = screen_name

            user_mentions.append(post_user_mention)
        return user_mentions

    def __convert_row_to_image_tags(self, row, post):
        image_tags = []
        photos = row["photos"]
        for photo in photos:
            image_tag = Image_Tags()

            image_tag.post_id = post.post_id
            image_tag.author_guid = post.author
            image_tag.media_path = photo

            image_tags.append(image_tag)
        return image_tags

    def __convert_row_to_post_citations(self, row, post):
        post_citations = []

        urls = row["urls"]
        for url in urls:
            post_citation = Post_citation()

            post_citation.post_id_from = post.post_id
            post_citation.post_id_to = "Unknown"
            post_citation.url_from = post.url
            post_citation.url_to = url

            post_citations.append(post_citation)
        return post_citations

    def __convert_row_to_terms_and_term_tweet_connections(self, row, post, term_descr_term_dict):
        hashtags = eval(row["hashtags"])

        terms = []
        term_tweet_connections = []
        for hashtag in hashtags:
            if hashtag not in term_descr_term_dict:
                term = self.__create_term(hashtag)
                term_descr_term_dict[hashtag] = term

                terms.append(term)
            else:
                term = term_descr_term_dict[hashtag]

            term_tweet_connection = Term_Tweet_Connection()

            term_tweet_connection.term_id = term.term_id
            term_tweet_connection.post_id = post.post_id
            term_tweet_connections.append(term_tweet_connection)

        return terms, term_tweet_connections

    def __convert_df_row_to_post(self, row, author):
        post = Post()

        post.domain = self._domain

        post_id = row["id"]
        post.post_id = post_id

        created_at = row["created_at"]
        post.created_at = created_at

        str_date = re.split("Jerusalem Daylight Time|Jerusalem Standard Time", created_at)[0].strip()
        date = str_to_date(str_date)
        post.date = date

        post_url = row["link"]
        post.url = post_url

        username = row["username"]
        post.author = username

        post.author_guid = author.author_guid

        post_guid = compute_post_guid(post_url, username, str_date)
        post.guid = post_guid

        content = row["tweet"]
        post.content = content

        language = row["language"]
        post.language = language

        post.original_tweet_importer_insertion_date = datetime.datetime.now()

        return post

    def insert_posts_csv_into_db(self):
        current_path = self._output_file_path + "crawled_tweets\\"
        current_path = os.path.join(current_path, "{0}-{1}\\".format(self._since, self._until))

        self.__next_term_id, terms = self._db.get_next_term_id()
        term_descr_term_dict = {term.description: term for term in terms}

        total_term_tweet_connections = []
        total_terms = []
        total_post_citations = []
        authors = []
        posts = []

        files = os.listdir(current_path)
        #files = [files[0]]

        for file in tqdm(files):

            df = pd.read_csv(current_path + file)

            for index, row in tqdm(df.iterrows()):
                #print(f"index={index}")
                author = self.__convert_df_row_to_author(row)
                authors.append(author)

                post = self.__convert_df_row_to_post(row, author)
                posts.append(post)

                terms, term_tweet_connections = self.__convert_row_to_terms_and_term_tweet_connections(row, post, term_descr_term_dict)

                total_terms = total_terms + terms
                total_term_tweet_connections = total_term_tweet_connections + term_tweet_connections

                post_citations = self.__convert_row_to_post_citations(row, post)
                total_post_citations = total_post_citations + post_citations

        self._db.addPosts(posts)
        self._db.addPosts(authors)
        self._db.addPosts(total_terms)
        self._db.addPosts(total_term_tweet_connections)
        self._db.addPosts(total_post_citations)