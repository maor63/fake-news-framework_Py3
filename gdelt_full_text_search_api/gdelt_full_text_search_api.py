
import urllib.request, urllib.parse, urllib.error
from preprocessing_tools.abstract_controller import AbstractController

class GDLET_Full_Text_Search_API(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._keywords = self._config_parser.eval(self.__class__.__name__, "keywords")
        self._specific_site = str(self._config_parser.get(self.__class__.__name__, "specific_site"))
        self._specific_language = str(self._config_parser.get(self.__class__.__name__, "specific_language"))
        self._maximal_returned_records = self._config_parser.get(self.__class__.__name__, "maximal_returned_records")

        self._prefix = "http://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch?&"
        self._api_parameters_seperator = '%20'


    def get_news_by_keywords(self, keywords=None):
        if keywords is None:
            keywords_seperated_for_api = self._api_parameters_seperator.join(self._keywords)
        else:
            keywords_seperated_for_api = self._api_parameters_seperator.join(keywords)

        query = self._prefix

        query += "query=" + keywords_seperated_for_api

        if self._specific_site is not '':
            query += self._api_parameters_seperator + "domain:" + self._specific_site

        if self._specific_language is not '':
            query += self._api_parameters_seperator + "sourcelang:" + self._specific_language

        query += "&output=urllist&dropdup=true&"

        query += "maxrows=" + self._maximal_returned_records


        # {0} = keywords
        # {1} = sort result by : rel,date
        # {2} = number of results
        #query = """http://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch?&query={0}%20sortby:{1}%20sourcelang:english&output=urllist&dropdup=true&maxrows={2}"""

        #"http://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch?&query=security%20domain:cnn.com%20sourcelang:english&output=urllist&dropdup=true&maxrows=100"

        results = urllib.request.urlopen(query)
        return results








