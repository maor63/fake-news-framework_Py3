import logging
from logging import config
from configuration.config_class import getConfig
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api
from DB.schema_definition import DB
from commons.commons import *
from preprocessing_tools.post_csv_exporter import PostCSVExporter
import os

class MissingVicoPostsRetriever():
    def __init__(self):
        config_parser = getConfig()
        logging.config.fileConfig(getConfig().get("DEFAULT", "logger_conf_file"))
        logger = logging.getLogger(getConfig().get("DEFAULT", "logger_name"))

        logger.info("Start Execution ... ")

        self._missing_retweets_not_retrived_from_vico_file_name = config_parser.get(self.__class__.__name__, "missing_retweets_not_retrived_from_vico_file_name")
        self._missing_tweets_not_retrived_from_vico_file_name = config_parser.get(self.__class__.__name__, "missing_tweets_not_retrived_from_vico_file_name")
        self._retweets_retrieved_from_vico_file_name = config_parser.get(self.__class__.__name__, "retweets_retrieved_from_vico_file_name")
        self._tweets_retrieved_from_vico_file_name = config_parser.get(self.__class__.__name__,
                                                                         "tweets_retrieved_from_vico_file_name")
        self._path = config_parser.get(self.__class__.__name__, "path")
        self._backup_path = config_parser.get(self.__class__.__name__, "backup_path")
        self._csv_header = config_parser.eval(self.__class__.__name__, "csv_header")
        self._csv_header_bad_actors_vico_retrieved_posts = config_parser.eval(self.__class__.__name__, "csv_header_bad_actors_vico_retrieved_posts")

        targeted_twitter_post_ids = config_parser.get("BadActorsCollector", "targeted_twitter_post_ids")
        self._targeted_twitter_post_ids = create_ids_from_config_file(targeted_twitter_post_ids)

        self._original_statuses = config_parser.eval(self.__class__.__name__, "original_statuses")


        self._csv_importer = PostCSVExporter()

        self._social_network_crawler = Twitter_Rest_Api()

        self._db = DB()
        self._db.setUp()

    def execute(self):
        '''
        id = 714718743973208064
        id = 3190956770
        timeline = self._social_network_crawler.get_timeline_by_user_id(id)
        x = 3
        '''
        #timelines = self.collect_bad_actors_not_retrieved_from_vico_timelines()
        #self.export_retweets_vico_not_retrieved(timelines)
        self.export_tweets_vico_not_retrieved()
        #self.export_tweets_retrieved_from_vico()
        #self.export_retweets_vico_retrieved()


    def export_retweets_vico_not_retrieved(self, bad_actors_timelines):
        #
        # A retweet is defined as a post that has no text of its own. It always starts with RT @creator.
        # If you reply to the tweet it is not defined as retweet.
        # A retweet from the timeline always has a retweeted_status object which includes the original status.
        # The retweet's text always starts with RT: @creator and the text of the user.
        #
        missing_retweets = []

        for timeline in bad_actors_timelines:
            missing_post = self.find_missing_retweet(timeline)
            if missing_post is not None:
                missing_retweets.append(missing_post)

        if len(missing_retweets) > 0:
            self.move_existing_file_to_backup(self._path, self._backup_path, self._missing_retweets_not_retrived_from_vico_file_name)
            missing_posts_content = self.create_missing_posts_content_for_csv(missing_retweets)
            full_path_file_name = self._path + self._missing_retweets_not_retrived_from_vico_file_name
            self._csv_importer.write_content_to_csv(missing_posts_content,
                                                    full_path_file_name, self._csv_header)


    def find_missing_retweet(self, timeline):
        for post in timeline:
            retweeted_status = post.retweeted_status
            if retweeted_status is not None:
                original_post_id = retweeted_status.id

                for post_id in self._targeted_twitter_post_ids:
                    if original_post_id == post_id:
                        return post
        return None

    def create_missing_posts_content_for_csv(self, missing_posts):
        missing_posts_content = []

        for missing_post in missing_posts:
            post_twitter_id = str(missing_post.id)
            missing_author_screen_name = missing_post.user.screen_name
            content = missing_post.text
            created_at = missing_post.created_at
            url = "http://twitter.com/"+ missing_author_screen_name + "/status/" + str(post_twitter_id)
            user_mentions = missing_post.user_mentions
            user_mention = user_mentions[0]

            original_author_twitter_id = str(user_mention.id)
            original_author_screen = user_mention.screen_name

            missing_post_content = [post_twitter_id, missing_author_screen_name, content, created_at, url, original_author_twitter_id, original_author_screen]
            missing_posts_content.append(missing_post_content)

        return missing_posts_content

    def create_missing_posts_content_for_csv(self, missing_posts):
        missing_posts_content = []

        for missing_post in missing_posts:
            post_twitter_id = str(missing_post.id)
            missing_author_screen_name = missing_post.user.screen_name
            content = missing_post.text
            created_at = missing_post.created_at
            url = "http://twitter.com/" + missing_author_screen_name + "/status/" + str(post_twitter_id)
            user_mentions = missing_post.user_mentions
            if len(user_mentions) > 0:
                user_mention = user_mentions[0]

                original_author_twitter_id = str(user_mention.id)
                original_author_screen = user_mention.screen_name

            else:
                urls = missing_post.urls
                original_url = urls[0]
                original_url = original_url.expanded_url
                relevant_part = original_url.split("https://twitter.com/", 1)
                screen_name_status_id = relevant_part[1].split("/status/", 1)
                original_author_twitter_id = str(screen_name_status_id[1])
                original_author_screen = screen_name_status_id[0]

            missing_post_content = [post_twitter_id, missing_author_screen_name, content, created_at, url,
                                        original_author_twitter_id, original_author_screen]
            missing_posts_content.append(missing_post_content)

        return missing_posts_content


    def move_existing_file_to_backup(self, original_path, backup_path, file_name):
        logging.info("move_existing_file_to_backup ")
        full_path_output_file = original_path + file_name
        if os.path.isfile(full_path_output_file):
            full_path_backup_output_file = backup_path + file_name
            if os.path.isfile(full_path_backup_output_file):
                os.remove(full_path_backup_output_file)
            os.rename(full_path_output_file, full_path_backup_output_file)

    def export_tweets_vico_not_retrieved(self):
        bad_actors_timelines = self.collect_bad_actors_timelines()

        missing_tweets = []

        for timeline in bad_actors_timelines:
            missing_post = self.find_missing_tweet(timeline)
            if missing_post is not None:
                missing_tweets.append(missing_post)

        if len(missing_tweets) > 0:
            self.move_existing_file_to_backup(self._path, self._backup_path, self._missing_tweets_not_retrived_from_vico_file_name)
            missing_posts_content = self.create_missing_posts_content_for_csv(missing_tweets)
            full_path_file_name = self._path + self._missing_tweets_not_retrived_from_vico_file_name
            self._csv_importer.write_content_to_csv(missing_posts_content,
                                                    full_path_file_name, self._csv_header)

    def find_missing_tweet(self, timeline):
        for post in timeline:
            retweeted_status = post.retweeted_status
            if retweeted_status is None:
                urls = post.urls
                if len(urls) > 0:
                    url = urls[0]
                    original_status_url = url.expanded_url
                    if original_status_url in self._original_statuses:
                        return post
        return None

    def collect_bad_actors_not_retrieved_from_vico_timelines(self):
        timelines = []
        bad_actors_not_found_by_vico_authors_ids = self._db.get_bad_actor_retweeters_not_retrieved_from_vico()
        for id in bad_actors_not_found_by_vico_authors_ids:
            timeline = self._social_network_crawler.get_timeline_by_user_id(id)
            if timeline is not None:
                timelines.append(timeline)
        return timelines

    def collect_bad_actors_timelines(self):
        timelines = []
        bad_actors_not_found_by_vico_authors_ids = self._db.get_bad_actor_ids()
        for id in bad_actors_not_found_by_vico_authors_ids:
            timeline = self._social_network_crawler.get_timeline_by_user_id(id)
            if timeline is not None:
                timelines.append(timeline)
        return timelines

    def export_retweets_vico_retrieved(self):
        retweets = self._db.get_bad_actors_retweets_retrieved_by_vico()
        self.move_existing_file_to_backup(self._path, self._backup_path,
                                          self._retweets_retrieved_from_vico_file_name)
        vico_retweets_content = self.create_bad_actors_posts_content_for_csv(retweets)
        full_path_file_name = self._path + self._retweets_retrieved_from_vico_file_name
        self._csv_importer.write_content_to_csv(vico_retweets_content,
                                                full_path_file_name, self._csv_header_bad_actors_vico_retrieved_posts)



    def create_bad_actors_posts_content_for_csv(self, retweets):
        retweets_content = []

        for retweet in retweets:
            post_id = str(retweet.post_id)
            author = retweet.author
            guid = retweet.guid
            title = retweet.title
            url = retweet.url
            date = retweet.date
            content = retweet.content
            domain = retweet.domain
            author_guid = retweet.author_guid

            retweet_content = [post_id, author, guid, title, url,
                                    date, content, domain, author_guid]
            retweets_content.append(retweet_content)

        return retweets_content

    def export_tweets_retrieved_from_vico(self):
        tweets = self._db.get_bad_actor_tweets_from_vico()
        self.move_existing_file_to_backup(self._path, self._backup_path,
                                          self._tweets_retrieved_from_vico_file_name)

        vico_tweets_content = self.create_bad_actors_posts_content_for_csv(tweets)
        full_path_file_name = self._path + self._tweets_retrieved_from_vico_file_name
        self._csv_importer.write_content_to_csv(vico_tweets_content,
                                                full_path_file_name, self._csv_header_bad_actors_vico_retrieved_posts)

    pass



