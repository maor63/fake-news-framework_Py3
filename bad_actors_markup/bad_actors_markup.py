# Created by aviade      
# Time: 24/05/2016 11:31

from preprocessing_tools.abstract_controller import AbstractController
from social_network_crawler.social_network_crawler import SocialNetworkCrawler
from datetime import datetime
from commons.consts import DB_Insertion_Type
from commons.commons import *
from preprocessing_tools.post_csv_exporter import PostCSVExporter
from configuration.config_class import getConfig
import os

class BadActorsMarkup(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._config_parser = getConfig()
        self._path = self._config_parser.get(self.__class__.__name__, "path")
        self._backup_path = self._config_parser.get(self.__class__.__name__, "backup_path")
        self._bad_actors_file_name = self._config_parser.get(self.__class__.__name__, "bad_actors_file_name")
        self._potential_good_actors_file_name = self._config_parser.get(self.__class__.__name__, "potential_good_actors_file_name")
        self._csv_header = self._config_parser.eval(self.__class__.__name__, "csv_header")

        self._vico_importer_bad_actors = []
        self._csv_importer = PostCSVExporter()


    def setUp(self):
        pass

    def execute(self,window_start=None):
        vico_importer_bad_actors = self.markup_vico_importer_bad_actors()
        self.export_bad_actors_csv_file()
        self.export_potential_good_actors_csv_file()

    def extract_vico_importer_bad_actors(self):
        logging.info("extract_vico_importer_bad_actors from db")
        self._vico_importer_bad_actors = self._db.get_vico_importer_bad_actors()

    def markup_vico_importer_bad_actors(self):
        logging.info("markup_vico_importer_bad_actors")
        self.extract_vico_importer_bad_actors()
        #self.save_vico_importer_bad_actors()
        return self._vico_importer_bad_actors

    def save_vico_importer_bad_actors(self):
        for vico_importer_bad_actor in self._vico_importer_bad_actors:
            current_date = get_current_time_as_string()
            vico_importer_bad_actor.bad_actors_markup_insertion_date = current_date

        self._db.addAuthors(self._vico_importer_bad_actors)

    def export_authors_to_csv_file(self, authors, file_name):

        self._csv_importer.write_content_to_csv(authors, file_name, self._csv_header)

    def export_bad_actors_csv_file(self):
        logging.info("extract_vico_importer_bad_actors")
        self.move_existing_file_to_backup(self._bad_actors_file_name)
        full_path_output_bad_actors_file = self._path + self._bad_actors_file_name
        bad_actors_content = self.create_authors_content_for_writer(self._vico_importer_bad_actors)
        self._csv_importer.write_content_to_csv(bad_actors_content, full_path_output_bad_actors_file, self._csv_header)

    def export_potential_good_actors_csv_file(self):
        logging.info("export_potential_good_actors_csv_file")
        self.move_existing_file_to_backup(self._potential_good_actors_file_name)
        potential_good_actors = self._db.get_vico_importer_potential_good_actors()


        full_path_output_potential_good_actors_file_name = self._path + self._potential_good_actors_file_name
        potential_good_actors_content = self.create_authors_content_for_writer(potential_good_actors)
        self._csv_importer.write_content_to_csv(potential_good_actors_content, full_path_output_potential_good_actors_file_name, self._csv_header)

    def move_existing_file_to_backup(self, file_name):
        logging.info("move_existing_file_to_backup " + file_name)
        full_path_output_file = self._path + file_name
        if os.path.isfile(full_path_output_file):
            full_path_backup_output_file = self._backup_path + file_name
            if os.path.isfile(full_path_backup_output_file):
                os.remove(full_path_backup_output_file)
            os.rename(full_path_output_file, full_path_backup_output_file)

    def create_authors_content_for_writer(self, authors):
        authors_content = []
        for author in authors:
            name = createunicodedata(author.name)
            logging.info("write_author: " + name)
            domain = createunicodedata(author.domain)
            author_guid = author.author_guid
            author_screen_name = createunicodedata(author.author_screen_name)
            author_full_name = createunicodedata(author.author_full_name)
            author_osn_id = author.author_osn_id
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

            author_content = [name, domain, author_guid, author_screen_name, author_full_name, author_osn_id,
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
                              author_sub_type]

            authors_content.append(author_content)

        return authors_content

if __name__ == "__main__":
    pass