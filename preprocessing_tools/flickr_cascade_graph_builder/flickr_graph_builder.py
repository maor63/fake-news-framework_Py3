
import datetime

import matplotlib.pyplot as plt
from scrapy.crawler import CrawlerProcess, Crawler, CrawlerRunner
import networkx as nx
from twisted.internet import reactor

from commons.commons import *
from DB.schema_definition import Post
from preprocessing_tools.abstract_controller import AbstractController
from preprocessing_tools.flickr_cascade_graph_builder.flicker_cascade_spider import FlickrCascadeSpider
import xml.etree.ElementTree as ET


class FlickrGraphBuilder(AbstractController):

    def __init__(self, db):
        super(FlickrGraphBuilder, self).__init__(db)
        config_eval = self._config_parser.eval
        self._photos_output_path = config_eval(self.__class__.__name__, "photos_output_path")
        self._input_flickr_photos_xmls_path = config_eval(self.__class__.__name__, "input_flickr_photos_xmls_path")
        self._photo_limit = config_eval(self.__class__.__name__, "photo_limit")

    def execute(self, window_start=None):
        posts = self.read_photos_as_posts()
        posts = self.filter_existing_photos(posts)[:self._photo_limit]

        posts_url = [post.url for post in posts]
        if posts_url != []:
            self.crawl_photos(posts_url)

            cascade_graphs_path = os.path.join(self._photos_output_path, 'cascade_graphs/')
            if not os.path.isdir(cascade_graphs_path):
                os.makedirs(cascade_graphs_path)

            failed_photos = [url.strip('/').split('/')[-1] for url in FlickrCascadeSpider.failed_urls]
            for photo_id in failed_photos:
                FlickrCascadeSpider.cascade_graphs_dict.pop(photo_id, None)

            print('{} photos failed because api key changed'.format(len(failed_photos)))
            self.generate_cascade_graphs_images(cascade_graphs_path)
            posts = [post for post in posts if post.post_osn_id not in failed_photos]
            self._db.add_author_connections(FlickrCascadeSpider.authors_connections)
            self._db.add_authors(FlickrCascadeSpider.authors)
            self._db.addPosts(posts)
            FlickrCascadeSpider.clean()
        else:
            print("########  all photos crawled  ##########")

    def filter_existing_photos(self, posts):
        posts_in_db = set(post.post_id for post in self._db.get_posts())
        posts = list([p for p in posts if p.post_id not in posts_in_db])
        return posts

    def read_photos_as_posts(self):
        posts = []
        for xml_name in os.listdir(self._input_flickr_photos_xmls_path):
            print("load {}".format(xml_name))
            posts += self.read_flicker_xml(os.path.join(self._input_flickr_photos_xmls_path, xml_name))
        return posts

    def crawl_photos(self, posts_url):
        FlickrCascadeSpider.set_photos_output_path(self._photos_output_path)
        crawler = Crawler(FlickrCascadeSpider, settings={
            'posts_url': posts_url
        })

        process = CrawlerProcess()
        process.crawl(crawler)
        process.start()
        pass

    def generate_cascade_graphs_images(self, cascade_graphs_path):
        for photo_id, graph in list(FlickrCascadeSpider.cascade_graphs_dict.items()):
            print('photo:', photo_id, 'nodes:', len(graph.nodes), 'edges:', len(graph.edges))
            labels = FlickrCascadeSpider.graph_nodes_labels[photo_id]
            labels = {name: label for name, label in list(labels.items()) if name in set(graph.nodes)}
            nx.draw(graph, labels=labels, with_labels=True, node_color='c')
            plt.title('graph {}, nodes: {}, edges: {}'.format(photo_id, len(graph.nodes), len(graph.edges)))
            plt.savefig(os.path.join(cascade_graphs_path, "photo_{}_cascade_graph.png".format(photo_id)), format="PNG")
            # plt.show()
            plt.clf()

    def read_flicker_xml(self, xml_path):
        tree = ET.parse(xml_path)
        photos = tree.getroot()
        return self.photos_to_posts(photos)

    def photos_to_posts(self, root):
        posts = []
        for i, photo in enumerate(root):
            print('\rconvert photos to posts {}/{}'.format(str(i+1), len(root)), end='')
            posts.append(self.photo_xml_to_post(photo))
        print()
        return posts

    def photo_xml_to_post(self, child):
        p = Post()
        p.title = str(child.find('title').text)
        p.url = str(child.find('urls').find('url').text)
        try:
            p.tags = ','.join(tag.text for tag in child.find('tags').findall('tag'))
        except:
            pass
        p.created_at = str(child.find('dates').get('posted'))
        p.date = datetime.datetime.fromtimestamp(int(p.created_at))
        p.author = str(child.find('owner').get('nsid'))
        p.domain = 'flickr'
        p.author_guid = compute_author_guid_by_author_name(p.author)
        p.retweet_count = int(child.find('comments').text)
        p.post_id = compute_post_guid(p.url, p.author, date_to_str(p.date))
        p.post_osn_id = str(child.get('id'))
        if child.find('labels') is not None:
            p.post_type = ','.join(tag.text for tag in child.find('labels').findall('label'))
        return p
