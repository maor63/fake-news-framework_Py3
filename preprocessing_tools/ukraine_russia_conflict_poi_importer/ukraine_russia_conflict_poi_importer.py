from commons.commons import *
from commons.method_executor import Method_Executor
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd
import datetime
from DB.schema_definition import UkraineRussiaConflictPOI
from tqdm import tqdm

class UkraineRussiaConflictPOIImporter(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._input_file_path = self._config_parser.get(self.__class__.__name__, "input_file_path")
        self._input_file_name = self._config_parser.get(self.__class__.__name__, "input_file_name")
        self._output_file_path = self._config_parser.get(self.__class__.__name__, "output_file_path")

    def execute(self, window_start=None):
        df = pd.read_csv(self._input_file_path + self._input_file_name)
        df["user_screen_name"] = df["screen_name"].apply(lambda s: s[1::])
        current_time = str(datetime.datetime.now())[0:11]
        df.to_csv(self._output_file_path + f"pois_ready_for_db_{current_time}.csv")

        self._save_POIs_DB(df)

    def _save_POIs_DB(self, df):
        pois = []
        for index, row in tqdm(df.iterrows()):
            user_screen_name = row['user_screen_name']
            poi_type = row['POI_type']
            original_screen_name = row['screen_name']
            url = row['URL']

            poi = UkraineRussiaConflictPOI()
            poi.user_screen_name = user_screen_name
            poi.poi_type = poi_type
            poi.url = url
            poi.original_screen_name = original_screen_name

            pois.append(poi)

        self._db.addPosts(pois)
        print("Done!!")







    # def parse_authors(self):
    #
    #     full_path_tsv_authors_file_name = self._tsv_input_files_path + self._tsv_authors_file_name
    #     authors = []
    #
    #     file = open(full_path_tsv_authors_file_name)
    #     logging.info("IMPORT CSV %s" % file)
    #     try:
    #         reader, num_of_records = self._get_reader_and_num_of_records(file, self._authors_header)
    #         i = 0
    #         for record_dict in reader:
    #             # if i != 0:
    #             author_name = record_dict["tumblog_id"]
    #             if author_name != 'NULL':
    #                 author = self._create_author_by_row(record_dict)
    #                 authors.append(author)
    #                 print(("author id: {0} was parsed successfully! ".format(author_name) + "{0}/{1}".format(i,
    #                                                                                                             num_of_records)))
    #             i += 1
    #         self._db.addAuthors(authors)
    #
    #     except:
    #         logging.exception("ERROR: can't read file: ")
    #     finally:
    #         file.close()
    #
    # def _create_author_by_row(self, record_dict):
    #     author = Author()
    #
    #     author_osn_id = self._convert_to_unicode_value(record_dict["tumblog_id"])
    #     author.author_osn_id = author_osn_id
    #     author.name = author_osn_id
    #
    #     author.domain = self._domain
    #     author.author_guid = compute_author_guid_by_author_name(author.name)
    #
    #     tumblr_blog_name = self._convert_to_unicode_value(record_dict["tumblr_blog_name"])
    #     author.author_screen_name = tumblr_blog_name
    #
    #     author.description = self._convert_to_unicode_value(record_dict["tumblr_blog_description"])
    #     created_time_epoch = self._convert_to_unicode_value(record_dict["created_time_epoch"])
    #     if created_time_epoch is not None:
    #         datetime, str_datetime = convert_epoch_timestamp_to_datetime(created_time_epoch)
    #         author.created_at = str_datetime
    #     else:
    #         author.created_at = self._set_start_date()
    #
    #     author.url = self._convert_to_unicode_value(record_dict["tumblr_blog_url"])
    #     author.protected = get_boolean_value(record_dict["is_private"])
    #     author.time_zone = self._convert_to_unicode_value(record_dict["timezone"])
    #     author.language = self._convert_to_unicode_value(record_dict["language"])
    #
    #     is_private = record_dict["is_private"]
    #     if is_private == "TRUE":
    #         author.protected = 1
    #     else:
    #         author.protected = 0
    #
    #     return author
    #
    # def parse_posts(self):
    #     full_path_tsv_posts_file_name = self._tsv_input_files_path + self._tsv_posts_file_name
    #     posts = []
    #     authors = []
    #
    #     file = open(full_path_tsv_posts_file_name)
    #     logging.info("IMPORT CSV %s" % file)
    #     try:
    #         i = 0
    #         self._post_dict = {}
    #         reader, num_of_records = self._get_reader_and_num_of_records(file, self._posts_header)
    #
    #         for record_dict in reader:
    #             post_id = record_dict["post_id"]
    #             tumblog_id = record_dict["tumblog_id"]
    #             if post_id != 'NULL' and tumblog_id != 'NULL':
    #                 post = self._create_post_by_row(record_dict)
    #                 reblogged_from_post_id = record_dict["reblogged_from_post_id"]
    #                 if reblogged_from_post_id != 'NULL':
    #                     reblogged_from_metadata = eval(record_dict["reblogged_from_metadata"])
    #                     post_citation = self._create_post_citation_by_row(reblogged_from_metadata)
    #                     posts.append(post_citation)
    #                     author = self._create_author_by_citation(reblogged_from_metadata)
    #                     authors.append(author)
    #                 posts.append(post)
    #                 print(("post id: {0} was parsed successfully".format(post_id) + "{0}/{1}".format(i, num_of_records)))
    #             i += 1
    #             print(("i = " + str(i)))
    #
    #         posts_to_save = list(self._post_dict.values())
    #         self._db.addPosts(posts_to_save)
    #         self._db.addAuthors(authors)
    #
    #     except Exception as e:
    #         print(e)
    #         logging.exception("ERROR: can't read file: ")
    #     finally:
    #         file.close()
    #
    # def _create_post_by_row(self, record_dict):
    #
    #     post = Post()
    #
    #     post_id = self._convert_to_unicode_value(record_dict["post_id"])
    #     post.post_osn_id = post_id
    #     post.post_id = post_id
    #
    #     author_name = self._convert_to_unicode_value(record_dict["tumblog_id"])
    #     post.author = author_name
    #
    #     post_short_url = self._convert_to_unicode_value(record_dict["post_short_url"])
    #     self._set_post_url(post_short_url, author_name, post)
    #
    #     post_creation_date = self._convert_to_unicode_value(record_dict["created_time_epoch"])
    #     post.created_at = post_creation_date
    #     if post_creation_date is not None:
    #         post_formatted_creation_date, str_post_formatted_creation_date = convert_epoch_timestamp_to_datetime(post_creation_date)
    #         post.date = post_formatted_creation_date
    #     else:
    #         str_post_formatted_creation_date = self._set_start_date()
    #
    #     post.guid = compute_post_guid(post.url,  author_name, str_post_formatted_creation_date)
    #     post.post_osn_guid = post.guid
    #
    #     post.title = self._convert_to_unicode_value(record_dict["post_title"])
    #
    #     post_content = record_dict["post_content"]
    #     if post_content != 'NULL':
    #         content = json.loads(post_content.decode("utf-8"))
    #         #content = eval(record_dict["post_content"])
    #         final_content = ""
    #         if 'title' in list(content.keys()):
    #             title = content['title']
    #             final_content += title
    #         if 'text' in list(content.keys()):
    #             text = content['text']
    #             final_content += text
    #         post.content = self._convert_to_unicode_value(final_content)
    #     post.domain = self._domain
    #     post.author_guid = compute_author_guid_by_author_name(author_name)
    #     post.post_type = self._convert_to_unicode_value(record_dict["post_type"])
    #     post.post_format = self._convert_to_unicode_value(record_dict["post_format"])
    #     post.reblog_key = self._convert_to_unicode_value(record_dict["post_reblog_key"])
    #     post.tags = self._convert_to_unicode_value(record_dict["post_tags"])
    #     post.state = self._convert_to_unicode_value(record_dict["post_state"])
    #
    #     if post.post_osn_id not in self._post_dict:
    #         self._post_dict[post.post_osn_id] = post
    #
    #     return post
    #
    #
    # def _convert_to_unicode_value(self, value):
    #     if value == "NULL":
    #         return None
    #     if isinstance(value, str):
    #         return str(value, 'utf-8')
    #     return value
    #
    # def parse_author_connections(self):
    #     full_path_tsv_author_connections_file_name = self._tsv_input_files_path + self._tsv_author_connections_file_name
    #     temp_author_connections = []
    #
    #     try:
    #         f = open(full_path_tsv_author_connections_file_name)
    #         logging.info("IMPORT CSV %s" % f)
    #         i = 0
    #         reader, num_of_records = self._get_reader_and_num_of_records(f, self._author_connections_header)
    #         for record_dict in reader:
    #             # if i != 0:
    #             temp_author_connection = self._create_temp_author_connection_by_row(record_dict)
    #             temp_author_connections.append(temp_author_connection)
    #             print(("temp_author_connection: {0} - {1} - {2} was parsed successfully ".format(temp_author_connection.source_author_osn_id,
    #                                                                                       temp_author_connection.destination_author_osn_id,
    #                                                                                       temp_author_connection.connection_type)
    #                   + "{0}/{1}".format(i, num_of_records)))
    #             i += 1
    #         self._db.add_author_connections(temp_author_connections)
    #
    #         self._db.convert_temp_author_connections_to_author_connections(self._domain)
    #         f.close()
    #     except:
    #         logging.exception("ERROR: can't read file: ")
    #     finally:
    #         pass
    #
    # def _create_temp_author_connection_by_row(self, record_dict):
    #     temp_author_connection = TempAuthorConnection()
    #
    #     temp_author_connection.source_author_osn_id = self._convert_to_unicode_value(record_dict["tumblog_id"])
    #     temp_author_connection.destination_author_osn_id = self._convert_to_unicode_value(record_dict["followed_tumblog_id"])
    #     temp_author_connection.connection_type = Connection_Type.FOLLOWER
    #
    #     activity_time_epoch = self._convert_to_unicode_value(record_dict["activity_time_epoch"])
    #     if activity_time_epoch is not None:
    #         datetime, str_datetime = convert_epoch_timestamp_to_datetime(activity_time_epoch)
    #         temp_author_connection.insertion_date = str_datetime
    #     else:
    #         temp_author_connection.insertion_date = self._set_start_date()
    #
    #     #temp_author_connection.insertion_date = self._convert_to_unicode_value(record_dict["activity_time_epoch"])
    #
    #     return temp_author_connection
    #
    # def _create_post_citation_by_row(self, reblogged_from_metadata):
    #     original_post = Post()
    #
    #     parent_post_id = reblogged_from_metadata["parent_post_id"]
    #     original_post.post_osn_id = parent_post_id
    #     original_post.post_id = parent_post_id
    #
    #
    #     parent_post_blog_id = reblogged_from_metadata["parent_post_blog_id"]
    #     original_post.author = parent_post_blog_id
    #     original_post.author_guid = compute_author_guid_by_author_name(parent_post_blog_id)
    #     original_post.domain = self._domain
    #     parent_post_short_url = self._convert_to_unicode_value(reblogged_from_metadata["parent_post_short_url"])
    #
    #     self._set_post_url(parent_post_short_url, parent_post_blog_id, original_post)
    #
    #     if parent_post_id not in self._post_dict:
    #         self._post_dict[parent_post_id] = original_post
    #     return original_post
    #
    #
    # def _set_post_url(self, post_url, parent_post_blog_id, post):
    #     if post_url is not None:
    #         post.url = post_url
    #     else:
    #         post.url = 'http://tmblr.co/' + 'parent_post_blog_id=' + parent_post_blog_id + 'index=' + str(self._url_index)
    #         self._url_index += 1
    #
    # def _create_author_by_citation(self, reblogged_from_metadata):
    #     author = Author()
    #     parent_post_blog_id = reblogged_from_metadata["parent_post_blog_id"]
    #     author.author_osn_id = parent_post_blog_id
    #     author.name = parent_post_blog_id
    #     author.author_guid = compute_author_guid_by_author_name(author.name)
    #
    #     parent_post_blog_name = self._convert_to_unicode_value(reblogged_from_metadata["parent_post_blog_name"])
    #     author.author_screen_name = parent_post_blog_name
    #     author.url = self._convert_to_unicode_value(reblogged_from_metadata["parent_post_short_url"])
    #
    #     author.domain = self._domain
    #     return author
    #
    # def _set_start_date(self):
    #     return str(self._start_date.strip("date('')"))
    #
    # def _get_reader_and_num_of_records(self, f, header):
    #     reader = csv.DictReader(f, fieldnames=header, delimiter='\t')
    #     rows = list(reader)
    #     num_of_records = len(rows)
    #
    #     f.seek(0)
    #     reader = csv.DictReader(f, fieldnames=header, delimiter='\t')
    #     return reader, num_of_records

