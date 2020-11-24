# -*- coding: utf-8 -*-

from DB.schema_definition import Post, AuthorConnection
from commons.method_executor import Method_Executor
import concurrent.futures
from preprocessing_tools.abstract_controller import AbstractController
import os
import time
import json
from datetime import datetime
from commons.commons import *
import parse
import requests
from bs4 import BeautifulSoup
import sys, traceback


class InstagramScraper(Method_Executor):
    ### More information about the commends at https://github.com/rarcega/instagram-scraper
    def __init__(self, db):
        super(InstagramScraper, self).__init__(db)
        self._download_path = self._config_parser.eval(self.__class__.__name__, "download_path")
        self._scraping_history_path = 'data/output/instagram_scrap_history.txt'
        self._limit = self._config_parser.eval(self.__class__.__name__, "limit")
        self._media_type = 'image'
        self._until = str_to_date('2018-12-30 00:00:00')
        self._until_timestamp = time.mktime(self._until.timetuple())

    def setUp(self):
        if not os.path.exists(self._download_path):
            os.makedirs(self._download_path)

    def _get_start_cursor_for_location(self, location_id):
        req = requests.get('https://www.instagram.com/explore/locations/{}/'.format(location_id))
        soup = BeautifulSoup(req.text, "html.parser")
        scripts = soup.find_all('script')
        script = list([script for script in scripts if 'window._sharedData = ' in script.text])[0]
        api_key_container = re.search('window._sharedData = \{.*\};', script.text).group(0)
        token_json = api_key_container.replace('window._sharedData = ', '').replace(';', '')
        j = json.loads(token_json)
        location_metadata = j['entry_data']['LocationsPage'][0]['graphql']
        end_cursor = self._get_cursor_from_data(location_metadata)
        return end_cursor

    def _get_cursor_from_data(self, metadata, query_type='location'):
        cursor_dict = metadata[('%s' % query_type)][('edge_%s_to_media' % query_type)]['page_info']
        if cursor_dict['has_next_page']:
            end_cursor = cursor_dict['end_cursor']
        else:
            end_cursor = None
        return end_cursor

    def retrive_posts_chunks_for_location(self, location_id, chunk_size=12):
        end_cursor = self._get_start_cursor_for_location(location_id)
        query_hash = '36bd0f2bf5911908de389b8ceaa3be6d'
        posts_count = 0
        download_path = os.path.join(self._download_path, str(location_id))
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        while end_cursor:
            # .format(location_id, end_cursor, chunk_size)

            query_vars = '{"id":"%s","first":%s,"after":"%s"}' % (str(location_id), chunk_size, end_cursor)
            res = requests.get(
                'https://www.instagram.com/graphql/query/?query_hash={}&variables={}'.format(query_hash, query_vars))
            json_res = json.loads(res.text)['data']
            instagram_posts = []
            for edge in json_res['location']['edge_location_to_media']['edges']:
                instagram_posts.append(edge['node'])
            end_cursor = self._get_cursor_from_data(json_res)
            instagram_posts = [p for p in instagram_posts if int(p['taken_at_timestamp']) > self._until_timestamp]
            if len(instagram_posts) == 0:
                break
            posts_count += len(instagram_posts)
            self._download_images(instagram_posts, download_path)
            print('location: {}, retreive {} posts, end_cursor: {}'.format(location_id, posts_count, end_cursor))
            yield instagram_posts
        print('location: {}, retreive {} posts, end_cursor: {}'.format(location_id, posts_count, end_cursor))

    def retrive_posts_chunks_for_hashtag(self, hashtag):
        end_cursor = self._config_parser.eval(self.__class__.__name__, "end_cursor")
        posts_count = 0
        folder_name = '#{}'.format(hashtag).decode('utf-8')
        download_path = os.path.join(self._download_path, folder_name)
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        # query_hash = '7dabc71d3e758b1ec19ffb85639e427b'
        QUERY_HASHTAG = 'https://www.instagram.com/graphql/query/?query_hash=ded47faa9a1aaded10161a2ff32abb6b&variables={0}'
        QUERY_HASHTAG_VARS = '{{"tag_name":"{0}","first":50,"after":"{1}"}}'

        break_count = 0
        while end_cursor is not None:
            params = QUERY_HASHTAG_VARS.format(hashtag, end_cursor)
            query = QUERY_HASHTAG.format(params)
            try:
                res = requests.get(query)
                json_res = json.loads(res.text)['data']
                instagram_posts = []
                for edge in json_res['hashtag']['edge_hashtag_to_media']['edges']:
                    instagram_posts.append(edge['node'])
                end_cursor = self._get_cursor_from_data(json_res, query_type='hashtag')
                instagram_posts = [p for p in instagram_posts if int(p['taken_at_timestamp']) > self._until_timestamp]
                if len(instagram_posts) == 0:
                    break
                posts_count += len(instagram_posts)
                self._download_images(instagram_posts, download_path)
                last_post_time = str(instagram_posts[-1]['taken_at_timestamp'])
                print('hashtag: {}, retreive {} posts, timestamp: {} end_cursor: {}'.format(hashtag, posts_count,
                                                                                            last_post_time, end_cursor))
                with open(os.path.join(self._download_path, '#{}_cursor_data.txt'.format(hashtag)), 'ab') as f:
                    f.write(','.join([hashtag, end_cursor, last_post_time]))
                break_count = 0
                yield instagram_posts
            except Exception as e:
                break_count += 1
                print(e)
                print('sleep 60 sec')
                time.sleep(60)
                if break_count >= 5:
                    break
        print('hashtag: {}, retreive {} posts, end_cursor: {}'.format(hashtag, posts_count, end_cursor))

    def _download_images(self, instagram_posts, download_path):
        def download_url(url):
            return requests.get(url).content

        def save_image(image_data, image_name):
            with open(os.path.join(download_path, '{}.jpg'.format(image_name)), 'wb') as f:
                f.write(image_data)

        image_urls = [p['display_url'] for p in instagram_posts]
        image_names = list(map(self._get_image_name_from_url, image_urls))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            images = executor.map(download_url, image_urls)
            for image_data, image_name in zip(images, image_names):
                save_image(image_data, image_name)

    def get_author_timeline_by_media_path(self):
        media_paths = self._db.get_authors_media_pats()
        user_names = [parse.parse('https://www.instagram.com/{}/', media_path)[0] for media_path in media_paths]
        self._crawl_users_timelines(user_names)
        self._add_timelines_to_db(user_names)

    def get_posts_by_hashtag(self):
        hashtag = self._config_parser.eval(self.__class__.__name__, "hashtag")
        posts = []
        for instagram_posts in self.retrive_posts_chunks_for_hashtag(hashtag):
            posts += [self._generate_post(instagram_post) for instagram_post in instagram_posts]
            earliest_post_date = min([p.date for p in posts] + [self._until])
            if self._until > earliest_post_date:
                break
            if len(posts) > 1000:
                self._db.add_posts_fast(posts)
                posts = []
        posts = [p for p in posts if p.date > self._until]
        self._db.add_posts_fast(posts)
        posts = []

    def fill_coments_for_posts(self):
        posts = self._db.get_instagram_posts_without_comments()
        BASE_URL = 'https://www.instagram.com/'
        QUERY_COMMENTS = BASE_URL + 'graphql/query/?query_hash=33ba35852cb50da46f5b5e889df7d159&variables={0}'
        QUERY_COMMENTS_VARS = '{{"shortcode":"{0}","first":50,"after":"{1}"}}'
        author_connections = []
        total_comments = []
        posts_count = len(posts)
        for i, post in enumerate(posts):
            end_cursor = ''
            break_count = 0
            while end_cursor is not None:
                try:

                    post_shortcode = parse.parse('https://www.instagram.com/p/{}/', post.url)[0]

                    params = QUERY_COMMENTS_VARS.format(post_shortcode, end_cursor)
                    query = QUERY_COMMENTS.format(params)

                    res = requests.get(query)
                    json_res = json.loads(res.text)['data']
                    if not json_res['shortcode_media']:
                        break

                    comments = []
                    for edge in json_res['shortcode_media']['edge_media_to_comment']['edges']:
                        comments.append(self._generate_comment(edge['node'], post))

                    total_comments += comments
                    author_connections += self._generate_post_comment_connections(post, comments)
                    if len(total_comments) > 10000:
                        self._db.add_posts_fast(total_comments)
                        self._db.add_entity_fast('author_connections', author_connections)
                        author_connections = []
                        total_comments = []

                    cursor_dict = json_res['shortcode_media']['edge_media_to_comment']['page_info']
                    if cursor_dict['has_next_page']:
                        end_cursor = cursor_dict['end_cursor']
                    else:
                        end_cursor = None

                except Exception as e:
                    break_count += 1
                    print(e)
                    self._db.add_posts_fast(total_comments)
                    self._db.add_entity_fast('author_connections', author_connections)
                    author_connections = []
                    total_comments = []
                    print(res.text)
                    print('sleep 120 sec')
                    time.sleep(120)
                    if break_count >= 5:
                        break
                print('\rretrive comments for post {}/{}, coments retrieved {}'.format(i, posts_count,
                                                                                       len(total_comments)), end='')
            print()
        self._db.add_posts_fast(total_comments)
        self._db.add_entity_fast('author_connections', author_connections)
        author_connections = []
        total_comments = []

    def get_posts_by_location_id(self):
        location_id = self._config_parser.eval(self.__class__.__name__, "location_id")
        posts = []
        for instagram_posts in self.retrive_posts_chunks_for_location(location_id):
            posts += [self._generate_post(instagram_post) for instagram_post in instagram_posts]
            earliest_post_date = min([p.date for p in posts] + [self._until])
            if self._until > earliest_post_date:
                break
            if len(posts) > 10000:
                self._db.add_posts_fast(posts)
                posts = []
        posts = [p for p in posts if p.date > self._until]
        self._db.add_posts_fast(posts)
        posts = []
        # dont work !!!
        # self._crawl_posts_by_location(location_id)
        # self._add_posts_and_comment_to_db(location_id, is_username=False)

    def _crawl_posts_by_location(self, location_id):
        'instagram-scraper  --location 213041503 --include-location -m 5 --profile-metadata -t image --comments'
        self._download_path = os.path.join(self._download_path, str(location_id))
        command_parts = [
            'instagram-scraper',
            '--location {}'.format(location_id),
            # '--comments',
            '--media-metadata',
            '--include-location',
            # '--retry-forever',
            '-t image',
            '--latest-stamps {}'.format(self._scraping_history_path),
            '--destination {}'.format(self._download_path),
        ]
        if self._limit:
            command_parts += ['--maximum {}'.format(self._limit)]
        print('run command: {}'.format(' '.join(command_parts)))
        os.system(' '.join(command_parts))
        print()

    def _crawl_posts_by_hashtag(self, hashtag):
        'instagram-scraper  --location 213041503 --include-location -m 5 --profile-metadata -t image --comments'
        self._download_path = os.path.join(self._download_path, str(hashtag))
        command_parts = [
            'instagram-scraper',
            '--tag {}'.format(hashtag),
            '--comments',
            '--media-metadata',
            '--include-location',
            # '--retry-forever',
            '-t image',
            # '--latest-stamps {}'.format(self._scraping_history_path),
            '--destination {}'.format(self._download_path),
        ]
        if self._limit:
            command_parts += ['--maximum {}'.format(self._limit)]
        print('run command: {}'.format(' '.join(command_parts)))
        os.system(' '.join(command_parts))
        print()

    def _crawl_users_timelines(self, user_names):
        command_parts = ['instagram-scraper {}'.format(','.join(user_names))]
        command_parts += [
            '--comments',
            '--media-types none',
            '--media-metadata',
            # '--retry-forever',
            '--latest-stamps {}'.format(self._scraping_history_path),
            '--destination {}'.format(self._download_path),
        ]
        if self._limit:
            command_parts += ['--maximum {}'.format(self._limit)]
        print('run command: {}'.format(' '.join(command_parts)))
        os.system(' '.join(command_parts))
        print()

    def _add_timelines_to_db(self, user_names):
        for user_name in user_names:
            self._add_posts_and_comment_to_db(user_name)

    def _add_posts_and_comment_to_db(self, json_name, is_username=True):
        posts = []
        author_connections = []
        with open(os.path.join(self._download_path, '{}.json'.format(json_name))) as json_file:
            user_data = json.load(json_file)
            for instagram_post in user_data['GraphImages']:
                post = self._generate_post(instagram_post)
                if is_username:
                    post.author = str(json_name)
                posts.append(post)
                comments = self._generate_comments(instagram_post, post)
                posts += comments
                author_connections += self._generate_post_comment_connections(post, comments)
                if len(posts) > 10000:
                    self._db.add_posts_fast(posts)
                    self._db.add_entity_fast('author_connections', author_connections)
                    posts = []
                    author_connections = []
        self._db.add_posts_fast(posts)
        self._db.add_entity_fast('author_connections', author_connections)

    def _generate_post(self, instagram_post):
        post = Post()
        post.author_guid = str(instagram_post['owner']['id'])
        post.date = datetime.datetime.fromtimestamp(instagram_post['taken_at_timestamp'])
        post.post_osn_id = instagram_post['id']
        try:
            post.content = instagram_post['edge_media_to_caption']['edges'][0]['node']['text']
        except:
            pass
        post.retweet_count = instagram_post['edge_media_to_comment']['count']
        post.favorite_count = instagram_post['edge_media_preview_like']['count']
        post.url = 'https://www.instagram.com/p/{}/'.format(instagram_post['shortcode'])
        image_names = []
        # for url in instagram_post['urls']:
        #     image_name_contaner = url.split('/')[-1]
        #     end = image_name_contaner.index('?')
        #     image_names.append(image_name_contaner[:end])
        image_name = self._get_image_name_from_url(instagram_post['display_url'])
        post.media_path = str(instagram_post['display_url'])
        post.post_format = '{}'.format(image_name)
        post.domain = 'Instagram'
        post.post_type = 'post'
        post.post_id = str(post.post_osn_id)
        return post

    def _get_image_name_from_url(self, url):
        image_name_container = url.split('/')[-1]
        end = image_name_container.index('?')
        image_name = image_name_container[:end]
        return image_name

    def _generate_comments(self, instagram_post, post):
        comments = []
        if 'data' in instagram_post['edge_media_to_comment']:
            for instagram_comment in instagram_post['edge_media_to_comment']['data']:
                comment = self._generate_comment(instagram_comment, post)
                comments.append(comment)
        return comments

    def _generate_comment(self, instagram_comment, post):
        comment = Post()
        comment.date = datetime.datetime.fromtimestamp(instagram_comment['created_at'])
        comment.post_osn_id = instagram_comment['id']
        comment.content = str(instagram_comment['text'])
        comment.author = str(instagram_comment['owner']['username'])
        comment.author_guid = str(instagram_comment['owner']['id'])
        comment.url = '{}{}/'.format(post.url, comment.post_osn_id)
        comment.domain = 'Instagram'
        comment.post_type = 'comment'
        comment.post_id = str(comment.post_osn_id)
        return comment

    def _generate_post_comment_connections(self, post, comments):
        author_connections = []
        for comment in comments:
            ac = AuthorConnection()
            ac.connection_type = 'post_comment_connection'
            ac.source_author_guid = post.post_id
            ac.destination_author_guid = comment.post_id
            author_connections.append(ac)
        return author_connections
