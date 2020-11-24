import subprocess
from configuration.config_class import getConfig
import logging
from preprocessing_tools.abstract_controller import AbstractController
from commons.commons import *


class AutotopicModelCreator(AbstractController):
    '''
    Creates an LDA model for each time window by running a copy of SOMWEB_LDA.R script.
    The model will be stored in AUTOTOPICS HOME DIR: SOMEC-code\autotopics
    Note: the R script assumes the required libraries are installed. In order to install new libraries, use install.packages(<package name>)
    via R.
    '''

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._script_name = 'autotopic_model_creation.R'

        self._autotopics_dir = self._config_parser.get(self.__class__.__name__, "autotopics_dir")
        self._db_name = self._config_parser.get("DB", "DB_name_prefix") + \
                        self._config_parser.get("DEFAULT", "social_network_name") + \
                        self._config_parser.get("DB", "DB_name_suffix")
        self._min_topic = self._config_parser.get(self.__class__.__name__, "min_topics")
        self._max_topic = self._config_parser.get(self.__class__.__name__, "max_topics")
        self.single_post_per_author = self._config_parser.get(self.__class__.__name__, "single_post_per_author")
        self._thresh_low = self._config_parser.get(self.__class__.__name__, "thresh_low")
        self._frequent_keywords_to_remove = self._config_parser.get(self.__class__.__name__,
                                                                    "frequent_keywords_to_remove")

        self._directory = self._autotopics_dir + "/tmp1/"

    def setUp(self):
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        self.generate_script()

    def execute(self, window_start):
        self.run_script()

    def generate_script(self):
        '''
        Generates the R script. The script is consist of two parts:
        * The configuration: C\P from the config.ini.
        * The script itself: C\P from the original script
        '''
        original_script_name = self._config_parser.get(self.__class__.__name__, "script_name")

        with open(self._script_name, 'w') as f:
            current_dir = os.getcwd().replace('\\', '//')
            f.write("setwd(\"" + current_dir + "//\")\n")
            f.write("setwd(\"" + self._autotopics_dir + "\")\n")
            f.write("date_start <- \"" + date_to_str(self._window_start) + "\" \n")
            f.write("date_end <- \"" + date_to_str(self._window_end) + "\" \n")
            db_path = current_dir + "//data//input//"
            f.write("sqlite_db <- \"" + db_path + self._db_name + "\" \n")
            f.write("K_MIN = " + str(self._min_topic) + "\n")
            f.write("K_MAX = " + str(self._max_topic) + " \n")
            f.write("single_post_per_author = " + self.single_post_per_author + "\n")
            f.write("domain = '" + self._domain + "'\n")
            f.write("thresh_low = " + self._thresh_low + "\n")
            f.write("frequent_keywords_to_remove <- c(" + self._frequent_keywords_to_remove + ")" + "\n")
            MicroblogOnly = ""
            if (self._domain == "Microblog"):
                MicroblogOnly = "TRUE"
            else:
                MicroblogOnly = "FALSE"
            f.write("MicroblogOnly = " + MicroblogOnly + "\n")

            with open(self._autotopics_dir + "/" + original_script_name, 'r') as original_script:
                for line in original_script:
                    f.write(line)
        f.close()

    def run_script(self):
        try:
            r_script_path = self._config_parser.get(self.__class__.__name__, "r_script_path")
            print("file name {0}".format(self._script_name))
            subprocess.call([r_script_path, self._script_name])
        except Exception as exc:
            logging.info(" An exception was caught when running autotopic_model_creation.R. \n "
                         " Model was NOT created. \n "
                         " Check that posts exist for the time window "+str(self._window_start)+" "+str(self._window_end))
