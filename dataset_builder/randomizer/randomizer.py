
from preprocessing_tools.abstract_controller import AbstractController

class Randomizer(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._min_number_of_posts_per_author = self._config_parser.eval(self.__class__.__name__, "min_number_of_posts_per_author")
        self._random_authors_requirements = self._config_parser.eval(self.__class__.__name__, "random_authors_requirements")

    def setUp(self):
        self._db.create_author_guid_num_of_posts_view()

        for random_authors_requirement_dict in self._random_authors_requirements:
            authors_table_field_name = random_authors_requirement_dict["field_name"]
            authors_table_value = random_authors_requirement_dict["value"]
            num_of_random_authors = random_authors_requirement_dict["num_of_random_authors"]

            self._db.randomize_authors(self._min_number_of_posts_per_author, self._domain, authors_table_field_name,
                                              authors_table_value, num_of_random_authors)


    def execute(self, window_start):
        pass