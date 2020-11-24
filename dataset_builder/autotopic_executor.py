import subprocess
from preprocessing_tools.abstract_controller import AbstractController


class AutotopicExecutor(AbstractController):
    '''
    Runs an adjusted R script that fills the following tables:
    1. author_topic_mapping : maps each AUTHOR GUID to the mean topic distribution about which the author wrote.
    2. author_topics : maps topic_id to the top k terms corresponding to this topic. according to the author_topic_mapping table.
    3. post_topics_mapping : maps each POST to the topic with the highest probability.
    4. post_topics : maps topic_id to the top k terms corresponding to this topic. according to the post_topics_mapping table.

    Note: the R script assumes the required libraries are installed. In order to install new libraries, use install.packages(<package name>)
    via R.
    '''

    def __init__(self, db):
        AbstractController.__init__(self, db)

        self._db_path = self._config_parser.get(self.__class__.__name__, "DB_path").replace('\\', '/')
        self._db_name = self._config_parser.get(self.__class__.__name__, "DB_name_prefix") + \
                        self._config_parser.get("DEFAULT", "social_network_name") + \
                        self._config_parser.get(self.__class__.__name__, "DB_name_suffix")

        self._single_post_per_author = self._config_parser.get(self.__class__.__name__, "single_post_per_author")
        self._lda_model_path = self._config_parser.get(self.__class__.__name__, "lda_model_path")
        self._k = self._config_parser.get(self.__class__.__name__, "k")
        self._script_path = self._config_parser.get(self.__class__.__name__, "script_path")

        self._script_name = 'autotopic_wrapper.R'

    def setUp(self):
        self.generate_script()

    def execute(self, window_start=None):
        self.run_script()

    def generate_script(self):
        '''
        Generates the R script. The script is consist of two parts:
        * The configuration: C\P from the config.ini.
        * The script itself: C\P from the original script
        '''

        with open(self._script_name, 'w') as f:
            f.write("dbPath <- \"" + self._db_path + "\"\n")
            f.write("dbName <- \"" + self._db_name + "\"\n")
            f.write("ldaModelPath <- \"" + self._lda_model_path + "\"\n")
            f.write("single_post_per_author <- " + self._single_post_per_author + " \n ")
            f.write("k <- " + self._k + "\n")
            f.write("domain = '" + self._domain + "'\n")
            with open(self._script_path, 'r') as original_script:
                for line in original_script:
                    f.write(line)
                f.flush()
        f.close()

    def run_script(self):
        r_script_path = self._config_parser.get(self.__class__.__name__, "r_script_path")
        subprocess.call([r_script_path, self._script_name])
