

from DB.schema_definition import Author
from commons.commons import compute_author_guid_by_author_name
from preprocessing_tools.post_importer import PostImporter
import pandas as pd

#####################################################################################################################
# This module is responsible to import CSV file which includes only Twitter author screen names to DB.
# It was created to import the dataset of screen names provided by Alon Bartal who pulished a paper for ASONAM 2018.
#####################################################################################################################
class TwitterAuthorScreenNamesImporter(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        self._path_to_file = self._config_parser.eval(self.__class__.__name__, "path_to_file")

    def execute(self, window_start):
        author_screen_names_df = pd.read_csv(self._path_to_file)
        author_screen_names = author_screen_names_df['author_screen_name'].tolist()

        authors = []
        for i, author_screen_name in enumerate(author_screen_names):
            author = Author()
            msg = "\rCreate author: [{0}/{1}]".format(i, len(author_screen_names))
            print(msg, end="")

            author.author_screen_name = author_screen_name
            author.name = author_screen_name


            author_guid = compute_author_guid_by_author_name(author_screen_name)
            author.author_guid = author_guid

            author.domain = self._domain

            authors.append(author)
        self._db.addPosts(authors)