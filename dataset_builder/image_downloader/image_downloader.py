
from preprocessing_tools.abstract_controller import AbstractController
import urllib.request, urllib.error, urllib.parse
import sys

__author__ = "Aviad Elyashar"

#########
#This module is responsible to download an image using given url and save it
# in designated folder and to update posts table.
#########
class Image_Downloader(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._path_for_downloaded_images = self._config_parser.eval(self.__class__.__name__, "path_for_downloaded_images")
        self._source_target_images_to_download_dicts = self._config_parser.eval(self.__class__.__name__, "source_target_images_to_download_dicts")

    def execute(self, window_start):
        for source_target_images_to_download_dict in self._source_target_images_to_download_dicts:
            source_table_name = source_target_images_to_download_dict["source_table_name"]
            source_id_field = source_target_images_to_download_dict["source_id_field"]
            targeted_url_field_name = source_target_images_to_download_dict["targeted_url_field_name"]
            source_where_clauses = source_target_images_to_download_dict["source_where_clauses"]

            target_table_name = source_target_images_to_download_dict["target_table_name"]
            target_field_name = source_target_images_to_download_dict["target_field_name"]

            source_tuples = self._db.get_records_by_id_targeted_field_and_table_name(source_id_field, targeted_url_field_name,
                                                                                     source_table_name, source_where_clauses)
            # covert because after some time - we get an error that connection of DB is closed: sqlite3.ProgrammingError: Cannot operate on a closed database. (python / sqlite)

            sources = list(source_tuples)

            objects_to_update = []
            if target_table_name == "posts":
                #post_id_post_dictionary
                object_id_object_dict = self._db.get_post_dictionary()
            else:
                # author_id_author_dictionary
                object_id_object_dict = self._db.get_author_dictionary()
            i = 0
            post_counter = 0
            for source_tuple in sources:
                i += 1
                # msg = "\r Num of downloaded images: {0}".format(i)
                # print(msg, end="")
                source_tuple_id = source_tuple[0]
                source_url = source_tuple[1]
                target_file = self._path_for_downloaded_images + source_tuple_id + ".jpg"

                if len(source_url) > 0:
                    try:
                        opener = urllib.request.build_opener()

                        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                        resource = opener.open(source_url)

                        output = open(target_file, "wb")
                        output.write(resource.read())
                        output.close()

                        object = object_id_object_dict[source_tuple_id]
                        setattr(object, target_field_name, source_tuple_id + ".jpg")

                        objects_to_update.append(object)
                        post_counter += 1
                        msg = "\r Num of downloaded images: {0}".format(post_counter)
                        print(msg, end="")

                        if post_counter == 1000:
                            self._db.addPosts(objects_to_update)
                            objects_to_update = []
                            post_counter = 0

                    except:
                        print("Unexpected error:", sys.exc_info()[0])

            self._db.addPosts(objects_to_update)

