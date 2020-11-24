from .abstract_controller import AbstractController
from commons import *
from configuration.config_class import getConfig
from scipy.stats import expon
import os.path
import csv
import urllib.request, urllib.error, urllib.parse
from commons.commons import *

class PostCSVExporter(AbstractController):
    '''
    For each post in each time windows, this module extract the following information:
    - post id
    - date
    - author_name [None if not available]
    - author_guid [None if not available]
    - url
    - title [None if not available]
    - content [None if not available]
    - domain [None if not available]
    - likes count [0 if not available]
    - shares count [0 if not available]
    - views count [0 if not available]
    - comment count [0 if not available]
    - sentiment [None if not available]
    - type [None if not available]
    - tfidf score [-0.01 if not available]
    - sum tfidf, max tfidf, avg boost score, sum boost score of the author [-0.01 if not available]
    - random score [generated according to the scores distribution]
    - topic id + topic text [None if not available]
    - # the post was referenced [0 if not available]
    - post's references [Empty entry if not available]

    To enable the module, add the following entry to the config file:
    [PostCSVExporter]
    file_name = <output_file_name>.csv
    '''


    def __init__(self):
        config_file = getConfig()
        self._path = config_file.get(self.__class__.__name__,"path")

    def execute(self,window_start):
        win_analyze = datetime.timedelta(seconds = int(getConfig().get("DEFAULT","window_analyze_size_in_sec")))
        window_end = window_start + win_analyze

        posts_data = self._db.get_posts_data(window_start, window_end)
        key_posts_score = self._db.get_key_posts_score(window_start, window_end)
        key_authors_score = self._db.get_key_authors_score(window_start, window_end)
        topics_data = self._db.get_topics_data(window_start, window_end)
        posts_references_count = self._db.get_reference_count(window_start, window_end)

        output = []
        for post_id in posts_data:
            post_data = []
            post_data = post_data + self.get_post_data(posts_data[post_id])
            post_data = post_data + self.get_post_score(key_posts_score, post_id)
            post_data = post_data + self.get_author_score(key_authors_score, post_id)
            post_data = post_data + self.sample_key_posts_score()
            post_data = post_data + self.get_topic_data(topics_data, post_id)
            post_data = post_data + self.get_posts_references_count(posts_references_count, post_id)
            post_data = post_data + self.get_post_references(post_id)
            output = output + [post_data]
        self.write_posts_to_csv(output)

    def get_post_data(self, post_data):
        '''
        :param post_data: a list containing the data regarding the post
        :return: post_data list where strings that might contain commas are surrounded by quotation marks.
        That will preserve the correct comma-separation structure of the csv file
        '''
        post_id = post_data[0]
        post_date = post_data[1]
        author_name = post_data[2]
        author_guid = post_data[3]
        url = "\"" + post_data[4].replace("\"", "\'") + "\""
        title =  post_data[5]
        if title is not None:
            title = "\"" + title.replace("\"", "\'") +"\""
        content = post_data[6]
        if content is not None:
            content = "\"" + post_data[6].replace("\"", "\'") + "\""
        domain = post_data[7]
        likes = post_data[8]
        shares = post_data[9]
        views = post_data[10]
        comment_count = post_data[11]
        sentiment = post_data[12]
        type = post_data[13]
        return [post_id, post_date, author_name, author_guid, url, title, content, domain, likes, shares, views, comment_count, sentiment, type]

    def get_posts_references_count(self, posts_citations_count, post_id):
         if post_id in posts_citations_count:
            return posts_citations_count[post_id][1:]
         else:
            return [0]

    def get_topic_data(self, topics_data, post_id):
        if post_id in topics_data:
            return topics_data[post_id][1:]
        else:
            return [None] * 2

    def get_post_score(self, key_posts_data, post_id):
        if post_id in key_posts_data:
                return key_posts_data[post_id][1:]
        else:
            return [-0.01]

    def get_author_score(self, key_authors_data, post_id):
        if post_id in key_authors_data:
            return key_authors_data[post_id][1:]
        else:
            return [-0.01] * 4

    def get_post_references(self, post_id):
        references = self._db.get_post_references(post_id)
        return ["|".join(references)]

    def write_posts_to_csv(self, posts):
        output_file = getConfig().get(self.__class__.__name__,"file_name")
        if not os.path.isfile(output_file):
            self.write_csv_header(output_file)
        with open(output_file, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for post in posts:
                p = [createunicodedata(p) for p in post]
                writer.writerow(p)

    def write_content_to_csv(self, contents, full_path_output_file, csv_header_properties):
        logging.info("write_authors_to_csv")

        if not os.path.isfile(full_path_output_file):
            self.write_csv_header(full_path_output_file, csv_header_properties)

        with open(full_path_output_file, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for content in contents:
                writer.writerow(content)

    def write_missing_posts_to_csv(self, missing_posts, full_path_output_file, csv_header_properties):
        logging.info("write_missing_posts_to_csv")
        # full_path_output_file = self._path + file_name

        if not os.path.isfile(full_path_output_file):
            self.write_csv_header(full_path_output_file, csv_header_properties)

        authors_content = self.create_authors_content_for_writer(authors)
        with open(full_path_output_file, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for author_content in authors_content:
                writer.writerow(author_content)


    def write_csv_header(self, output_file, header_properties):
        with open(output_file, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header_properties)
            '''
            writer.writerow(["name", "domain", "author_guid", "author_screen_name", "author_full_name",
                             "author_twitter_id", "description", "created_at", "statuses_count",
                             "followers_count", "favourites_count", "friends_count", "listed_count", "language",
                             "profile_background_color", "profile_background_tile", "profile_banner_url",
                             "profile_image_url", "profile_link_color", "profile_sidebar_fill_color", "profile_text_color",
                             "default_profile", "contributors_enabled", "default_profile_image", "geo_enabled", "protected",
                             "location", "notifications", "time_zone", "url", "utc_offset", "verified", "is_suspended_or_not_exists", "author_type",
                             "bad_actors_collector_insertion_date", "xml_importer_insertion_date", "vico_dump_importer_insertion_date",
                             "missing_data_complementor_insertion_date", "bad_actors_markup_insertion_date"])
            '''
    def sample_key_posts_score(self):
        return list(expon.rvs(loc=0, scale=0.0737837097238, size=1))


    def create_authors_content_for_writer(self, authors):
        authors_content = []
        for author in authors:
            name = createunicodedata(author.name)
            logging.info("write_author: " + name)
            domain = createunicodedata(author.domain)
            author_guid = author.author_guid
            author_screen_name = createunicodedata(author.author_screen_name)
            author_full_name = createunicodedata(author.author_full_name)
            author_twitter_id = author.author_twitter_id
            description = author.description.encode('utf8')
            # description = createunicodedata(author.description)
            created_at = author.created_at
            statuses_count = author.statuses_count
            followers_count = author.followers_count
            favourites_count = author.favourites_count
            friends_count = author.friends_count
            listed_count = author.listed_count
            language = createunicodedata(author.language)
            profile_background_color = author.profile_background_color
            profile_background_tile = author.profile_background_tile
            profile_banner_url = author.profile_banner_url
            profile_image_url = author.profile_image_url
            profile_link_color = author.profile_link_color
            profile_sidebar_fill_color = author.profile_sidebar_fill_color
            profile_text_color = author.profile_text_color
            default_profile = author.default_profile
            contributors_enabled = author.contributors_enabled
            default_profile_image = author.default_profile_image
            geo_enabled = author.geo_enabled
            protected = author.protected
            location = createunicodedata(author.location)
            notifications = author.notifications
            time_zone = createunicodedata(author.time_zone)
            url = createunicodedata(author.url)
            utc_offset = author.utc_offset
            verified = author.verified
            is_suspended_or_not_exists = author.is_suspended_or_not_exists
            author_type = author.author_type
            bad_actors_collector_insertion_date = author.bad_actors_collector_insertion_date
            xml_importer_insertion_date = author.xml_importer_insertion_date
            vico_dump_insertion_date = author.vico_dump_insertion_date
            missing_data_complementor_insertion_date = author.missing_data_complementor_insertion_date
            bad_actors_markup_insertion_date = author.bad_actors_markup_insertion_date
            mark_missing_bad_actor_retweeters_insertion_date = author.mark_missing_bad_actor_retweeters_insertion_date
            author_sub_type = author.author_sub_type

            author_content = [name, domain, author_guid, author_screen_name, author_full_name, author_twitter_id,
                                description, created_at, statuses_count, followers_count, favourites_count, friends_count,
                                listed_count, language, profile_background_color, profile_background_tile,
                                profile_banner_url,
                                profile_image_url, profile_link_color, profile_sidebar_fill_color, profile_text_color,
                                default_profile,
                                contributors_enabled, default_profile_image, geo_enabled, protected, location,
                                notifications,
                                time_zone, url, utc_offset, verified, is_suspended_or_not_exists, author_type,
                                bad_actors_collector_insertion_date,
                                xml_importer_insertion_date, vico_dump_insertion_date,
                                missing_data_complementor_insertion_date, bad_actors_markup_insertion_date, mark_missing_bad_actor_retweeters_insertion_date,
                                author_sub_type
                              ]

            authors_content.append(author_content)

        return authors_content
