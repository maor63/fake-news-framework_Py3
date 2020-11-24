import requests

from commons.commons import *
import json
import os
import re
from urllib.request import urlretrieve, urlopen
from urllib.request import Request
from urllib.request import urlopen

import networkx as nx
import parse
import scrapy
from bs4 import BeautifulSoup
from scrapy import Spider
from operator import itemgetter
from DB.schema_definition import Author, AuthorConnection
from commons.commons import *

user_data = 'https://api.flickr.com/services/rest?per_page=20&page=1&extras=can_addmeta%2Ccan_comment%2Ccan_download%2Ccan_share%2Ccontact%2Ccount_comments%2Ccount_faves%2Ccount_views%2Cdate_taken%2Cdate_upload%2Cdescription%2Cicon_urls_deep%2Cisfavorite%2Cispro%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cowner_datecreate%2Cpath_alias%2Crealname%2Crotation%2Csafety_level%2Csecret_k%2Csecret_h%2Curl_c%2Curl_f%2Curl_h%2Curl_k%2Curl_l%2Curl_m%2Curl_n%2Curl_o%2Curl_q%2Curl_s%2Curl_sq%2Curl_t%2Curl_z%2Cvisibility%2Cvisibility_source%2Co_dims%2Cpubliceditability&get_user_info=1&jump_to=&user_id={}&viewerNSID=&method=flickr.people.getPhotos&csrf=&api_key=16985e6b3d581ee4d7ca908ba4cf88c9&format=json&hermes=1&hermesClient=1&reqId=3cce0e93&nojsoncallback=1'
comments = 'https://api.flickr.com/services/rest?photo_id={0}&sort=date-posted-desc&extras=icon_urls&expand_bbml=1&use_text_for_links=1&secure_image_embeds=1&bbml_need_all_photo_sizes=1&primary_photo_longest_dimension=405&offset={2}&limit={1}&viewerNSID=&method=flickr.photos.comments.getList&csrf=&api_key={3}&format=json&hermes=1&hermesClient=1&reqId=3cce0e93&nojsoncallback=1'
likes = 'https://api.flickr.com/services/rest?photo_id={0}&extras=can_addmeta%2Ccan_comment%2Ccan_download%2Ccan_share%2Ccontact%2Ccount_comments%2Ccount_faves%2Ccount_views%2Cdate_taken%2Cdate_upload%2Cdescription%2Cicon_urls_deep%2Cisfavorite%2Cispro%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cowner_datecreate%2Cpath_alias%2Crealname%2Crotation%2Csafety_level%2Csecret_k%2Csecret_h%2Curl_c%2Curl_f%2Curl_h%2Curl_k%2Curl_l%2Curl_m%2Curl_n%2Curl_o%2Curl_q%2Curl_s%2Curl_sq%2Curl_t%2Curl_z%2Cvisibility%2Cvisibility_source%2Co_dims%2Cpubliceditability&per_page=1000&page=1&hermes=1&sort=date_asc&viewerNSID=&method=flickr.photos.getFavorites&csrf=&api_key={1}&format=json&hermesClient=1&reqId=88a72bc2&nojsoncallback=1'
galleries = 'https://api.flickr.com/services/rest?photo_id=830593398&extras=can_addmeta%2Ccan_comment%2Ccan_download%2Ccan_share%2Ccontact%2Ccount_comments%2Ccount_faves%2Ccount_views%2Cdate_taken%2Cdate_upload%2Cdescription%2Cicon_urls_deep%2Cisfavorite%2Cispro%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cowner_datecreate%2Cpath_alias%2Crealname%2Crotation%2Csafety_level%2Csecret_k%2Csecret_h%2Curl_c%2Curl_f%2Curl_h%2Curl_k%2Curl_l%2Curl_m%2Curl_n%2Curl_o%2Curl_q%2Curl_s%2Curl_sq%2Curl_t%2Curl_z%2Cvisibility%2Cvisibility_source%2Co_dims%2Cpubliceditability%2Cdatecreate%2Cdate_activity%2Ceighteenplus%2Cinvitation_only%2Cneeds_interstitial%2Cnon_members_privacy%2Cpool_pending_count%2Cprivacy%2Cmember_pending_count%2Cicon_urls%2Cdate_activity_detail%2Cowner_name%2Cpath_alias%2Crealname%2Csizes%2Curl_m%2Curl_n%2Curl_q%2Curl_s%2Curl_sq%2Curl_t%2Curl_z%2Curl_c%2Curl_h%2Curl_k%2Curl_l%2Curl_z%2Cneeds_interstitial&primary_photo_extras=url_sq%2C%20url_t%2C%20url_s%2C%20url_m%2C%20needs_interstitial&get_all_galleries=1&no_faves_context=1&per_type_limit=6&get_totals=1&sort=date-desc&viewerNSID=&method=flickr.photos.getAllContexts&csrf=&api_key=16985e6b3d581ee4d7ca908ba4cf88c9&format=json&hermes=1&hermesClient=1&reqId=3cce0e93&nojsoncallback=1'
follows = 'https://api.flickr.com/services/rest?viewerNSID=&csrf=&api_key={1}&format=json&hermes=1&hermesClient=1&reqId=3cce0e93&nojsoncallback=1&per_page=1000&page={2}&get_user_info=1&jump_to=&user_id={0}&method=flickr.contacts.getPublicList'
photo_methods_base = 'https://api.flickr.com/services/rest?viewerNSID&csrf&photo_id={0}&api_key={1}&format=json&nojsoncallback=1&method=%s'
photo_data = photo_methods_base % 'flickr.photos.getInfo'
photo_url = photo_methods_base % 'flickr.photos.getSizes'


# def retry_when_api_change(parse_fn):
#     def retry(self, response):
#         try:
#             parse_fn(self, response)
#         except:
#             print "replace api key"
#             post_id = response.meta['photo_id']
#             self.authors = list(filter(lambda a: a.domain != post_id, self.authors))
#             self.authors_connections = list(filter(lambda a: a.type != post_id, self.authors_connections))
#             if post_id in self.cascade_graphs_dict:
#                 del self.cascade_graphs_dict[post_id]
#                 del self.graph_nodes_labels[post_id]
#
#     return retry


class FlickrCascadeSpider(Spider):
    name = "flickr_cascade_spider"

    scraper_url = 'https://www.flickr.com/'
    api_key = ""
    output_path = 'photos/'
    cascade_graphs_dict = {}
    graph_nodes_labels = {}
    authors = []
    authors_connections = []
    posts_url = []
    user_follows_dict = {}
    failed_urls = []
    photo_id_to_url = {}

    @classmethod
    def clean(cls):
        cls.cascade_graphs_dict = {}
        cls.graph_nodes_labels = {}
        cls.authors = []
        cls.authors_connections = []

    @classmethod
    def change_api_key(cls):
        cls.api_key = cls.get_current_api_key()

    @classmethod
    def set_photos_output_path(cls, output_path):
        cls.output_path = output_path

    @classmethod
    def get_current_api_key(cls):
        request = requests.get('https://www.flickr.com/')
        soup = BeautifulSoup(request.text, "html.parser")
        scripts = soup.find_all('script')
        script = list([script for script in scripts if 'root.YUI_config.flickr.api.site_key = ' in script.text])[0]
        api_key_container = re.search('root.YUI_config.flickr.api.site_key = ".*";', script.text).group(0)
        api_key = parse.parse('root.YUI_config.flickr.api.site_key = "{}";', api_key_container)[0]
        return api_key

    def start_requests(self):
        self.api_key = self.get_current_api_key()
        self.photo_id_to_url = {url.strip('/').split('/')[-1]: url for url in self.settings.getlist('posts_url')}
        self.photos_output_path = self.init_photos_dir_path('photos/')
        for post_url in self.settings.getlist('posts_url'):
            yield scrapy.Request(url=post_url, callback=self.parse)

    def init_photos_dir_path(self, photos_path):
        photos_output_path = os.path.join(self.output_path, photos_path)
        if not os.path.isdir(photos_output_path):
            os.makedirs(photos_output_path)
        return photos_output_path

    def clean_photo_data(self, failure):
        post_id = failure.response.meta['photo_id']
        self.authors = list([a for a in self.authors if a.domain != post_id])
        self.authors_connections = list([a for a in self.authors_connections if a.type != post_id])
        del self.cascade_graphs_dict[post_id]
        del self.graph_nodes_labels[post_id]

    def parse(self, response):
        assert isinstance(response, scrapy.http.Response)
        url = response.url
        photo_author_alias, photo_id = url.strip('/').split('/')[-2:]
        print((photo_author_alias, photo_id))
        self.cascade_graphs_dict[photo_id] = nx.DiGraph()
        self.graph_nodes_labels[photo_id] = {}
        print(('work on {}/{} photos'.format(len(self.cascade_graphs_dict), len(self.settings.getlist('posts_url')))))
        # request = scrapy.Request(url=self.get_photo_url_api(photo_id), callback=self.download_photo)
        request = scrapy.Request(url=self.get_photo_data_api(photo_id), callback=self.get_photo_data)
        request.meta['photo_id'] = photo_id
        yield request

    def download_photo(self, response):
        jsonresponse = json.loads(response.text)
        photo_original_url = jsonresponse['sizes']['size'][-1]['source']
        photo_id = response.meta['photo_id']
        photo_destination = os.path.join(self.photos_output_path, "{}.jpg".format(photo_id))
        urlretrieve(photo_original_url, photo_destination)

    def get_photo_data(self, response):
        jsonresponse = json.loads(response.text)
        if jsonresponse.get('code', 200) == 100:
            self.failed_urls.append(self.photo_id_to_url[response.meta['photo_id']])
        else:
            owner_id = jsonresponse['photo']['owner']["nsid"]
            author = self.owner_to_author(jsonresponse)

            self.authors.append(author)
            photo_id = jsonresponse['photo']['id']
            self.cascade_graphs_dict[photo_id].add_nodes_from([owner_id])
            self.graph_nodes_labels[photo_id][owner_id] = 'owner'

            request = scrapy.Request(url=self.likes_api(photo_id), callback=self.parse_likes)
            request.meta['owner_id'] = owner_id
            request.meta['photo_id'] = photo_id
            yield request

    def parse_likes(self, response):
        assert isinstance(response, scrapy.http.Response)
        jsonresponse = json.loads(response.text)
        if jsonresponse.get('code', 200) == 100:
            self.failed_urls.append(self.photo_id_to_url[response.meta['photo_id']])
        else:
            likers = jsonresponse['photo']['person']
            photo_id = response.meta['photo_id']

            authors = self.likers_to_authors(likers, photo_id)
            likers_data = {liker['nsid']: liker['favedate'] for liker in likers}
            self.cascade_graphs_dict[photo_id].add_nodes_from(likers_data)
            self.graph_nodes_labels[photo_id].update({author_id: 'like' for author_id in likers_data})

            self.authors.extend(authors)
            photo_id = response.meta['photo_id']
            request = scrapy.Request(url=self.get_comments_api(photo_id), callback=self.parse_photo_comments)
            request.meta['likers_data'] = likers_data
            request.meta['owner_id'] = response.meta['owner_id']
            request.meta['photo_id'] = response.meta['photo_id']
            yield request

    def parse_photo_comments(self, response):
        assert isinstance(response, scrapy.http.Response)
        jsonresponse = json.loads(response.text)
        if jsonresponse.get('code', 200) == 100:
            self.failed_urls.append(self.photo_id_to_url[response.meta['photo_id']])
        else:
            photo_id = response.meta['photo_id']
            owner_id = response.meta['owner_id']
            comments = jsonresponse['comments']['comment']

            authors = self.commenters_to_authors(comments, photo_id)
            commenters_data = {comment['author']: comment['datecreate'] for comment in comments}
            self.cascade_graphs_dict[photo_id].add_nodes_from(commenters_data)
            self.graph_nodes_labels[photo_id].update({author_id: 'comment' for author_id in commenters_data})
            self.authors.extend(authors)

            likers_data = response.meta['likers_data']
            # self.save_node_labels(commenters_data, likers_data, owner_id, photo_id)
            likers_data.update(commenters_data)
            authors_data = likers_data
            authors_ids = sorted(authors_data, key=lambda id: authors_data[id])
            print(authors_ids)
            for i, author_id in enumerate(authors_ids):
                predecessor_author_ids = [owner_id] + authors_ids[:i]
                follows_api = self.get_follows_api(author_id)
                yield self.add_follows_edge(author_id, set(), follows_api, photo_id, predecessor_author_ids)

    def get_user_follows(self, response):
        assert isinstance(response, scrapy.http.Response)
        jsonresponse = json.loads(response.text)
        if jsonresponse.get('code', 200) == 100:
            self.failed_urls.append(self.photo_id_to_url[response.meta['photo_id']])
        else:
            photo_id = response.meta['photo_id']
            follower_id = response.meta['author_id']
            if 'contacts' in jsonresponse:
                follows = jsonresponse['contacts'].get('contact', [])
                total_follows = jsonresponse['contacts'].get('total', 0)
                follows_ids = response.meta['follows'] | {follow['nsid'] for follow in follows}
                # self.authors_connections.extend(self.create_follow_connections(follower_id, follows))
            else:
                follows_ids = response.meta['follows']
                total_follows = len(follows_ids)

            nodes = response.meta['nodes']
            if follower_id in self.user_follows_dict:
                follows_ids = self.user_follows_dict[follower_id]
            edges = [(node, follower_id) for node in nodes if node in follows_ids]
            if int(total_follows) <= len(follows_ids) or len(edges) == len(nodes) or follower_id in self.user_follows_dict:
                self.authors_connections.extend(self.create_cascade_edges(edges, photo_id))
                self.cascade_graphs_dict[photo_id].add_edges_from(edges)
                self.user_follows_dict[follower_id] = follows_ids
            else:
                next_page = str(int(jsonresponse['contacts']['page']) + 1)
                follows_api = self.get_follows_api(follower_id, next_page)
                yield self.add_follows_edge(follower_id, follows_ids, follows_api, photo_id, nodes)

    def owner_to_author(self, jsonresponse):
        author = Author()
        author.name = str(jsonresponse['photo']['owner']['username'])
        author.author_full_name = str(jsonresponse['photo']['owner']['realname'])
        author.domain = str(jsonresponse['photo']['id'])
        author.created_at = str(jsonresponse['photo']['dates']["posted"])
        author.author_osn_id = str(jsonresponse['photo']['owner']['nsid'])
        author.author_guid = compute_author_guid_by_author_name(author.author_osn_id)
        author.location = str(jsonresponse['photo']['owner']['location'])
        author.author_type = 'owner'
        return author

    def likers_to_authors(self, likers, photo_id):
        return [self.liker_to_author(liker, photo_id) for liker in likers]

    def liker_to_author(self, liker, photo_id):
        author = Author()
        author.name = str(liker['username'])
        author.author_full_name = str(liker.get('realname', ""))
        author.domain = str(photo_id)
        author.created_at = str(liker["favedate"])
        author.author_osn_id = str(liker['nsid'])
        author.author_guid = compute_author_guid_by_author_name(author.author_osn_id)
        author.author_type = 'like'
        return author

    def commenters_to_authors(self, comments, photo_id):
        authors = []
        for commenter in comments:
            author = self.commenter_to_author(commenter, photo_id)
            authors.append(author)
        return authors

    def commenter_to_author(self, commenter, photo_id):
        author = Author()
        author.name = str(commenter['authorname'])
        author.author_screen_name = str(commenter['path_alias'])
        author.author_full_name = str(commenter.get('realname', ""))
        author.url = str(commenter['permalink'])
        author.domain = str(photo_id)
        author.created_at = str(commenter["datecreate"])
        author.author_osn_id = str(commenter['author'])
        author.author_guid = compute_author_guid_by_author_name(author.author_osn_id)
        author.author_type = 'comment'
        return author

    def save_node_labels(self, commenters_data, likers_data, owner_id, photo_id):
        self.graph_nodes_labels[photo_id].update({author_id: 'comment' for author_id in commenters_data})
        self.graph_nodes_labels[photo_id].update({author_id: 'like' for author_id in likers_data})
        self.graph_nodes_labels[photo_id][owner_id] = 'owner'

    def create_cascade_edges(self, edges, photo_id):
        author_connections = []
        for node, follower_id in edges:
            author_connection = self.create_cascade_edge(follower_id, node, photo_id)
            author_connections.append(author_connection)
        return author_connections

    def create_cascade_edge(self, follower_id, node, photo_id):
        author_connection = AuthorConnection()
        author_connection.source_author_guid = compute_author_guid_by_author_name(follower_id)
        author_connection.destination_author_guid = compute_author_guid_by_author_name(node)
        author_connection.connection_type = str(photo_id)
        author_connection.weight = 1.0
        return author_connection

    def create_follow_connections(self, follower_id, follows):
        author_connections = []
        for follow_data in follows:
            author_connection = self.create_follow_connection(follow_data, follower_id)
            author_connections.append(author_connection)
        return author_connections

    def create_follow_connection(self, follow, follower_id):
        author_connection = AuthorConnection()
        author_connection.source_author_guid = compute_author_guid_by_author_name(follower_id)
        author_connection.destination_author_guid = compute_author_guid_by_author_name(follow['nsid'])
        author_connection.connection_type = 'follow'
        author_connection.weight = 1.0
        return author_connection

    def add_follows_edge(self, author_id, current_follows, follows_api, photo_id, predecessor_author_ids):
        request = scrapy.Request(url=follows_api, callback=self.get_user_follows)
        request.meta['author_id'] = author_id
        request.meta['nodes'] = predecessor_author_ids
        request.meta['photo_id'] = photo_id
        request.meta['follows'] = current_follows
        return request

    def get_comments_api(self, photo_id, comment_limit='200', offset='0'):
        return comments.format(photo_id, comment_limit, offset, self.api_key)

    def get_photo_data_api(self, photo_id):
        return photo_data.format(photo_id, self.api_key)

    def get_follows_api(self, user_id, page='1'):
        return follows.format(user_id, self.api_key, page)

    def likes_api(self, photo_id):
        return likes.format(photo_id, self.api_key)

    def get_photo_url_api(self, photo_id):
        return photo_url.format(photo_id, self.api_key)

# import matplotlib.pyplot as plt
#
# process = CrawlerProcess()
# process.crawl(FlickrCascadeSpider)
# process.start()
# for photo_id, graph in FlickrCascadeSpider.cascade_graphs_dict.items():
#     print('nodes:', len(graph.nodes), 'edges:', len(graph.edges))
#     labels = FlickrCascadeSpider.graph_nodes_labels[photo_id]
#     labels = {name: label for name, label in labels.items() if name in set(graph.nodes)}
#     nx.draw(graph, labels=labels, with_labels=True)
#     plt.title('graph {}, nodes: {}, edges: {}'.format(photo_id, len(graph.nodes), len(graph.edges)))
#     plt.savefig("photo_{}_cascade_graph.png".format(photo_id), format="PNG")
#     plt.show()
