import csv
import logging

from selenium.webdriver.chrome.options import Options
from missing_data_complementor.missing_data_complementor import MissingDataComplementor
from preprocessing_tools.post_importer import PostImporter
from twitter_crawler.generic_twitter_crawler import Generic_Twitter_Crawler
import selenium


class Twitter_Spam_Dataset_Importer(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        self._db = db
        self._twitter_crawler = Generic_Twitter_Crawler(db)
        self._missing_data_complementor = MissingDataComplementor(db)
        self._original_tsv_location = self._config_parser.eval(self.__class__.__name__, "original_tsv_location")
        self._limit_per_crawl = self._config_parser.eval(self.__class__.__name__, "limit_per_crawl")
        self._post_label_dict = {}
        self._start_len = len(list(self._post_label_dict.keys()))

        self._num_of_rows_in_dataset = 0
        self._post_author_to_label_dict = {}
        chrome_options = Options()
        chrome_options.add_argument("--headless") # to make the process run in background
        self._web_driver = selenium.webdriver.Chrome(executable_path=r'vendors\chromedriver\chromedriver.exe',chrome_options=chrome_options)


        # statistics vars
        self._counter = 0
        self._more_than_one = 0
        self._exactly_one = 0

        self._loggin_processed = 0

    def execute(self, window_start):
        self._tsv_to_dict()
        num_of_crawls = self._num_of_rows_in_dataset / self._limit_per_crawl
        num_of_crawls = num_of_crawls + 1
        for i in range(num_of_crawls):
            self._crawl_for_post_id_and_author_id(i)
            self._fill_and_save_posts_and_authors()
        self._web_driver.close()
        self._db.fill_author_type_by_post_type()
        logging.info("Exactly one: "+str(self._exactly_one))
        logging.info("More than one: "+str(self._more_than_one))

    def _tsv_to_dict(self):
        f = open(self._original_tsv_location)
        logging.info("IMPORT CSV %s" % f)
        try:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                post_content = row[0]
                post_label = row[1]
                self._post_label_dict[post_content] = post_label
        except Exception as exc:
            logging.ERROR("Error Importing from tsv: "+exc)
        finally:
            f.close()
            self._num_of_rows_in_dataset = len(list(self._post_label_dict.keys()))
            logging.info("Finished importing csv")

    def _crawl_for_post_id_and_author_id(self, index):
        items_found = 0
        items_looked_for = 0
        upper_limit = min(index + self._limit_per_crawl, len(list(self._post_label_dict.keys())))

        self._start_len = len(list(self._post_label_dict.keys()))
        for post_content in list(self._post_label_dict.keys()):
            logging.info("working on post "+str(self._counter)+" from: "+str(self._start_len))
            self._counter =self._counter +1
            if items_looked_for == upper_limit:
                return
            if post_content == '':
                continue
            if len(post_content.split(' '))<4: #to filter too common sentences
                continue
            clean_str = self._clean_post_content(post_content)
            post_author_tuple = self._get_user_post_by_content(clean_str)
            if post_author_tuple != (None, None):
                logging.info("found and marked "+str(self._loggin_processed)+" authors")
                self._post_author_to_label_dict[post_author_tuple] = self._post_label_dict[post_content]
                items_found = items_found +1
                self._loggin_processed = self._loggin_processed + 1
            items_looked_for = items_looked_for + 1
            self._post_label_dict.pop(post_content, None)
        print("should have processed "+str(items_found)+" posts")

    def _clean_post_content(self, content):
        clean_str = content.replace("@user", "")
        clean_str = clean_str.replace("link", "")
        clean_str = clean_str.replace("...", "")
        return clean_str

    def _get_user_post_by_content(self, post_content):
        post_id = None
        writer_id = None
        url = self._create_url(post_content)
        try:
            self._web_driver.get(url)
            content = self._web_driver.find_elements_by_class_name("content") # if wish to add more than one post
            num_of_tweets = len(content)
            if (num_of_tweets>1):
                self._more_than_one = self._more_than_one +1
                return (post_id, writer_id)
            else:
                self._exactly_one = self._exactly_one +1
            logging.info(str(num_of_tweets))
            post_id, writer_id = self.get_post_id_and_writer_id_from_selenium_content(content)
        except Exception as exc:
            logging.info("got Exception at selenium work: "+exc)
        finally:
            return (post_id,writer_id)

    def get_post_id_and_writer_id_from_selenium_content(self, content):
        content = content[0]
        headers = content.find_element_by_class_name('stream-item-header')
        header_link = headers.find_element_by_tag_name('a')
        writer_id = header_link.get_attribute('data-user-id')
        time = headers.find_element_by_class_name("time")
        post_link = time.find_element_by_tag_name("a")
        post_id_list = post_link.get_attribute('href').split('/')
        post_id = post_id_list[len(post_id_list) - 1]
        return post_id, writer_id

    def _create_url(self, post_content):
        query = self._create_query(post_content)
        url = "https://twitter.com/search?f=tweets&q="+query +"&src=typd"
        return url

    def _create_query(self, content):
        return content.replace(' ', '%20')

    def _fill_and_save_posts_and_authors(self):
        for post_id_author_id in list(self._post_author_to_label_dict.keys()):
            try:
                self._twitter_crawler.retrieve_and_save_data_from_twitter_by_post_id(post_id_author_id[0], self._post_author_to_label_dict[post_id_author_id])
                self._post_author_to_label_dict.pop(post_id_author_id, None)
            except Exception as e:
                print(e)
                logging.info("error in the twitter api at " + str(post_id_author_id[0]))
        self._twitter_crawler.commit_db()
