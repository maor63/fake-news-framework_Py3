

import http.client
import socket
import threading
import urllib.parse
from time import sleep

from DB.schema_definition import *
from preprocessing_tools.post_importer import PostImporter


class PostCitationCreator(PostImporter):
    def __init__(self, db):
        PostImporter.__init__(self, db)
        self._max_number_of_threads = self._config_parser.eval(self.__class__.__name__, "max_number_of_threads")
        self._shorten_url_expended_url_dict = {}
        self._destination_url_source_url_dict = {}
        self.resolved_urls = []

    def execute(self, window_start=None):

        posts = self._db.get_posts_filtered_by_domain(self._domain)
        urls = self._extract_urls_from_posts_content(posts)

        self._expand_urls(urls)

        posts_refs_dicts = self._create_post_refs_dicts()
        references = self.fromPostsRefsDictListToRefsList(posts_refs_dicts)
        self._add_references_to_db(references)

    def _extract_urls_from_posts_content(self, posts):
        urls = []
        print("Extract all urls from posts content")
        [urls.extend(self._extract_urls_from_post_content(post)) for post in posts]
        return urls

    def _extract_urls_from_post_content(self, post):
        urls = self._extract_urls_from_post(post)
        urls_for_expansion = []
        for destination_url in urls:
            if post.url is not None and destination_url is not None and len(destination_url) > 13:
                urls_for_expansion.append(destination_url)
                source_url = post.url
                self._destination_url_source_url_dict[destination_url] = source_url
        return urls_for_expansion

    def _expand_urls(self, urls):
        chunks = split_into_equal_chunks(urls, self._max_number_of_threads)
        threads = []

        i = 1
        for url_chunk in chunks:
            print("Url Chunk {0}/{1}".format(str(i), len(urls) / self._max_number_of_threads))
            i += 1
            for url in url_chunk:
                threads.append(threading.Thread(target=self._try_expand_shorten_url, args=(url,)))

            for t in threads:
                t.start()
            for t in threads:
                t.join()
            threads = []

    def _add_references_to_db(self, references):
        self._db.addReferences(references)

    def _create_post_refs_dicts(self):
        'clumsy implementation for using post importer'
        inner_posts_refs_dicts = []
        posts_refs_dicts = []
        for short_url in self._shorten_url_expended_url_dict:
            refDictItem = {}
            source_url = self._destination_url_source_url_dict[short_url]
            destination_url = self._shorten_url_expended_url_dict[short_url]
            refDictItem.update({'urlfrom': str(source_url)})
            refDictItem.update({'urlto': str(destination_url)})
            inner_posts_refs_dicts.append(refDictItem)
        posts_refs_dicts.append(inner_posts_refs_dicts)
        return posts_refs_dicts

    def _get_post_references(self, post):
        urls = self._extract_urls_from_post(post)

        postRefsDictList = []
        for ref in urls:
            if post.url is not None and ref is not None and len(ref) > 13:
                refDictItem = {}
                from_url = post.url
                if ref in self._shorten_url_expended_url_dict:
                    to_url = self._shorten_url_expended_url_dict[ref]
                else:
                    break
                refDictItem.update({'urlfrom': str(from_url)})
                refDictItem.update({'urlto': str(to_url)})
                postRefsDictList.append(refDictItem)
        return postRefsDictList

    def _extract_urls_from_post(self, post):
        content = str(post.content)
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        return urls

    def _try_expand_shorten_url(self, shorten_url):
        thread_name = threading.current_thread().name
        print("{0} : try expand {1}".format(thread_name, shorten_url))
        try:
            expanded_url = self.expand_url_with_redirection(shorten_url)

            active_threads = threading.activeCount()
            if (not expanded_url.startswith("https://t.co")) and (not expanded_url.startswith("http://t.co")):
                print("{0} : expand suc {1} ; active threads {2}".format(thread_name, expanded_url, active_threads))
                self._shorten_url_expended_url_dict[shorten_url] = expanded_url
        except socket.timeout as e:
            print(str(e) + " timeout error: " + shorten_url)
        except socket.error as e:
            print(str(e) + " broken url error: " + shorten_url)

    def expand_url_with_redirection(self, url):
        temp_link = url
        while True:
            link = self._unshorten_url(temp_link)
            if link == temp_link or (not link.startswith("http")):
                break
            temp_link = link
        return temp_link

    def _unshorten_url(self, url):
        parsed = urllib.parse.urlparse(url)
        http_connection = http.client.HTTPConnection(parsed.netloc, timeout=5)
        http_connection.request('HEAD', parsed.path)
        response = http_connection.getresponse()
        if response.status / 100 == 3 and response.getheader('Location'):
            return response.getheader('Location')
        else:
            print("expand suc from unshorten {0} ;".format(url))
            return url

