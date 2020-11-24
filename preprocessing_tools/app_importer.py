# encoding: utf-8
# need to be added to the system


import codecs
from os import listdir
from DB.schema_definition import *
from preprocessing_tools.post_importer import PostImporter


class AppImporter(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        config_parser = getConfig()
        self.start_date = config_parser.eval("DEFAULT", "start_date")
        self.end_date = config_parser.eval("DEFAULT", "end_date")
        self._data_folder = self._config_parser.eval(self.__class__.__name__, "data_folder")
        self._bad_actor_threshold = self._config_parser.eval(self.__class__.__name__, "bad_actor_threshold")
        self._optional_classes = self._config_parser.eval(self.__class__.__name__, "optional_classes")
        self._author_classify_dict = {}

    def execute(self, window_start=None):
        self._read_apps_from_folder()
        self._insert_posts_and_authors_to_db()

    def _read_apps_from_folder(self):
        all_apps = listdir(self._data_folder)
        i = 1
        self._author_classify_dict = {}
        for app in all_apps:
            if i % 100 == 0:
                msg = "\rImport app [{0}/{1}]".format(i, len(all_apps))
                print(msg, end="")
            i += 1
            self._parse_app_data_to_posts_and_authors(app)

    def _parse_app_data_to_posts_and_authors(self, app):
        self._extract_author_to_author_classify_dict(app)
        self._extract_posts_data_to_listdic(app)

    def _extract_posts_data_to_listdic(self, app):
        f = codecs.open(self._data_folder + app, "r", encoding='CP1252')
        rows = f.readlines()
        for row in rows:
            try:
                # guid = generate_random_guid()
                self._coverte_row_to_post_and_save(app, row)
            except:
                self.logger.error("Cant encode the post:{0}".format(app))
                pass
        f.close()

    def _coverte_row_to_post_and_save(self, app, row):
        clean_content, post_dict = self._parse_row_into_post(app, row)
        if not clean_content == "":
            self._add_to_posts_dicts(post_dict)

    def _parse_row_into_post(self, app, row):
        guid_start = app.find('_') + 1
        author_guid = str(app[guid_start:len(app) - 4])
        clean_content = self._get_clean_content_from_row(row)
        guid = str(compute_post_guid(str(clean_content), str(author_guid), str(self.start_date)))
        post_dict = {}
        post_dict["author_guid"] = author_guid
        post_dict["content"] = clean_content
        post_dict["date"] = str(self.start_date)
        post_dict["guid"] = guid
        post_dict["author"] = author_guid
        post_dict["references"] = ""
        post_dict["domain"] = self._domain
        post_dict["url"] = self._social_network_url + guid
        post_dict["author_osn_id"] = author_guid
        return clean_content, post_dict

    def _add_to_posts_dicts(self, post_dict):
        self._listdic.append(post_dict.copy())

    def _get_clean_content_from_row(self, row):
        return str(cleaner(row[:-1]))

    def _extract_author_to_author_classify_dict(self, app):
        author_type_start = app.find('_')
        author_guid = app[author_type_start + 1:len(app) - 4]
        if int(app[:author_type_start]) > self._bad_actor_threshold:
            # Suppose to be bad_actor
            self._author_classify_dict[author_guid] = str(self._optional_classes[1])
        else:
            # Suppose to be good_actor
            self._author_classify_dict[author_guid] = str(self._optional_classes[0])

    def _insert_posts_and_authors_to_db(self):
        print("Insert posts to DB")
        self.insertPostsIntoDB()
        print("Insert Authors")
        self._db.insert_or_update_authors_from_posts(self._domain, self._author_classify_dict, {})
