# Created by aviade
# Time: 31/03/2016 09:15

import logging
import os
import platform

import sqlalchemy
from configuration.config_class import getConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import aliased
from sqlalchemy import event
from sqlalchemy.sql.operators import ColumnOperators
from sqlalchemy import Column, func, and_, or_, not_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Boolean, Integer, Unicode, FLOAT
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.expression import *
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from commons.commons import *
from commons.consts import DB_Insertion_Type, Author_Type, Author_Connection_Type
import re
import itertools
from commons.consts import Domains, Social_Networks
import pandas as pd
import csv
from collections import defaultdict
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import mapper
from itertools import chain

Base = declarative_base()

configInst = getConfig()

dialect_name = getConfig().get("DB", "dialect_name")

exec('import ' + dialect_name)

exec('from ' + dialect_name + ' import DATETIME')

dt = eval(dialect_name).DATETIME(
    storage_format="%(year)04d-%(month)02d-%(day)02d %(hour)02d:%(minute)02d:%(second)02d",
    regexp=r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})",
)

domain = getConfig().get("DEFAULT", "domain")


class Author(Base):
    __tablename__ = 'authors'

    name = Column(Unicode, primary_key=True)
    domain = Column(Unicode, primary_key=True)
    author_guid = Column(Unicode, primary_key=True)

    author_screen_name = Column(Unicode, default=None)
    author_full_name = Column(Unicode, default=None)
    author_osn_id = Column(Unicode, default=None)
    description = Column(Unicode, default=None)
    created_at = Column(Unicode, default=None)
    statuses_count = Column(Integer, default=None)
    followers_count = Column(Integer, default=None)
    favourites_count = Column(Integer, default=None)
    friends_count = Column(Integer, default=None)
    listed_count = Column(Integer, default=None)
    language = Column(Unicode, default=None)
    profile_background_color = Column(Unicode, default=None)
    profile_background_tile = Column(Unicode, default=None)
    profile_banner_url = Column(Unicode, default=None)
    profile_image_url = Column(Unicode, default=None)
    profile_link_color = Column(Unicode, default=None)
    profile_sidebar_fill_color = Column(Unicode, default=None)
    profile_text_color = Column(Unicode, default=None)
    default_profile = Column(Unicode, default=None)
    contributors_enabled = Column(Unicode, default=None)
    default_profile_image = Column(Unicode, default=None)
    geo_enabled = Column(Unicode, default=None)
    protected = Column(Boolean, default=None)
    location = Column(Unicode, default=None)
    notifications = Column(Unicode, default=None)
    time_zone = Column(Unicode, default=None)
    url = Column(Unicode, default=None)
    utc_offset = Column(Unicode, default=None)
    verified = Column(Unicode, default=None)
    is_suspended_or_not_exists = Column(dt, default=None)

    # Tumblr fields
    default_post_format = Column(Unicode, default=None)
    likes_count = Column(Integer, default=None)
    allow_questions = Column(Boolean, default=False)
    allow_anonymous_questions = Column(Boolean, default=False)
    image_size = Column(Integer, default=None)

    media_path = Column(Unicode, default=None)

    author_type = Column(Unicode, default=None)
    bad_actors_collector_insertion_date = Column(Unicode, default=None)
    xml_importer_insertion_date = Column(Unicode, default=None)
    vico_dump_insertion_date = Column(Unicode, default=None)
    missing_data_complementor_insertion_date = Column(Unicode, default=None)
    bad_actors_markup_insertion_date = Column(Unicode, default=None)
    mark_missing_bad_actor_retweeters_insertion_date = Column(Unicode, default=None)
    author_sub_type = Column(Unicode, default=None)
    timeline_overlap_insertion_date = Column(Unicode, default=None)
    original_tweet_importer_insertion_date = Column(Unicode, default=None)

    def __repr__(self):
        return "<Author(name='%s',domain='%s',author_guid='%s', statuses_count='%s')>" % (
            self.name, self.domain, self.author_guid, self.statuses_count)


class PostAuthorsConnections(Base):
    __tablename__ = 'post_authors_connections'
    post_id = Column(Unicode, primary_key=True)
    user_id = Column(Unicode, primary_key=True)

class AuthorConnection(Base):
    __tablename__ = 'author_connections'

    source_author_guid = Column(Unicode, primary_key=True)
    destination_author_guid = Column(Unicode, primary_key=True)
    connection_type = Column(Unicode, primary_key=True)
    weight = Column(FLOAT, default=0.0)
    insertion_date = Column(Unicode, default=None)

    def __repr__(self):
        return "<AuthorConnection(source_author_guid='%s', destination_author_guid='%s', connection_type='%s', " \
               "weight='%s', insertion_date='%s')>" % (self.source_author_guid, self.destination_author_guid,
                                                       self.connection_type, self.weight, self.insertion_date)


class TempAuthorConnection(Base):
    __tablename__ = 'temp_author_connections'

    source_author_osn_id = Column(Unicode, primary_key=True)
    destination_author_osn_id = Column(Unicode, primary_key=True)
    connection_type = Column(Unicode, primary_key=True)
    weight = Column(FLOAT, default=0.0)
    insertion_date = Column(Unicode, default=None)

    def __repr__(self):
        return "<TempAuthorConnection(source_author_osn_id='%s', destination_author_osn_id='%s', connection_type='%s', " \
               "weight='%s', insertion_date='%s')>" % (self.source_author_osn_id, self.destination_author_osn_id,
                                                       self.connection_type, self.weight, self.insertion_date)


class PostRetweeterConnection(Base):
    __tablename__ = 'post_retweeter_connections'

    post_osn_id = Column(Integer, primary_key=True)
    retweeter_twitter_id = Column(Integer, primary_key=True)
    connection_type = Column(Unicode, primary_key=True)
    insertion_date = Column(Unicode, default=None)

    def __repr__(self):
        return "<Post_Retweeter_Connection(post_twitter_id='%s', retweeter_twitter_id='%s', connection_type='%s')>" % (
            self.post_osn_id, self.retweeter_twitter_id, self.connection_type)


class PostUserMention(Base):
    __tablename__ = 'post_user_mentions'

    post_guid = Column(Integer, primary_key=True)
    user_mention_twitter_id = Column(Integer, primary_key=True)
    user_mention_screen_name = Column(Unicode, default=None)

    def __repr__(self):
        return "<PostUserMention(post_guid='%s', user_mention_twitter_id='%s', user_mention_screen_name='%s')>" % (
            self.post_guid, self.user_mention_twitter_id, self.user_mention_screen_name)


class Post(Base):
    __tablename__ = 'posts'

    post_id = Column(Unicode, primary_key=True, index=True)
    author = Column(Unicode, default=None)
    guid = Column(Unicode, default=None)
    title = Column(Unicode, default=None)
    url = Column(Unicode, default=None)
    date = Column(dt, default=None)
    content = Column(Unicode, default=None)
    description = Column(Unicode, default=None)
    is_detailed = Column(Boolean, default=True)
    is_LB = Column(Boolean, default=False)
    is_valid = Column(Boolean, default=True)
    domain = Column(Unicode, primary_key=True, default=None)
    author_guid = Column(Unicode, default=None)

    media_path = Column(Unicode, default=None)

    # keywords = Column(Unicode, default=None)
    # paragraphs = Column(Unicode, default=None)
    post_osn_guid = Column(Unicode, default=None)
    post_type = Column(Unicode, default=None)
    post_format = Column(Unicode, default=None)
    reblog_key = Column(Unicode, default=None)
    tags = Column(Unicode, default=None)
    is_created_via_bookmarklet = Column(Boolean, default=None)
    is_created_via_mobile = Column(Boolean, default=None)
    source_url = Column(Unicode, default=None)
    source_title = Column(Unicode, default=None)
    is_liked = Column(Boolean, default=None)
    post_state = Column(Unicode, default=None)

    post_osn_id = Column(Integer, default=None)
    retweet_count = Column(Integer, default=None)
    favorite_count = Column(Integer, default=None)
    # reply_count = Column(Integer, default=None)
    # language = Column(Unicode, default=None)
    created_at = Column(Unicode, default=None)
    xml_importer_insertion_date = Column(Unicode, default=None)
    timeline_importer_insertion_date = Column(Unicode, default=None)
    original_tweet_importer_insertion_date = Column(Unicode, default=None)

    def __repr__(self):
        return "<Post(post_id='%s', guid='%s', title='%s', url='%s', date='%s', content='%s', author='%s', is_detailed='%s',  is_LB='%s',domain='%s',author_guid='%s')>" % (
            self.post_id, self.guid, self.title, self.url, self.date, self.content, self.author, self.is_detailed,
            self.is_LB, self.domain, self.author_guid)


class Post_citation(Base):
    __tablename__ = 'post_citations'

    post_id_from = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    post_id_to = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    url_from = Column(Unicode, index=True)  # need to be deleted do not use it
    url_to = Column(Unicode, index=True)  # need to be deleted do not use it

    def __repr__(self):
        return "<Post_citation(post_id_from='%s', post_id_to='%s', url_from='%s', url_to='%s')>" % (
            self.post_id_from, self.post_id_to, self.url_from, self.url_to)


class Target_Article(Base):
    __tablename__ = 'target_articles'

    post_id = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    author_guid = Column(Unicode, ForeignKey('posts.author_guid', ondelete="CASCADE"), primary_key=True)
    title = Column(Unicode, default=None)
    description = Column(Unicode, default=None)
    keywords = Column(Unicode, default=None)

    def __repr__(self):
        return "<TargetArticle(post_id='%s', author_guid='%s', title='%s', description='%s', keywords='%s')>" % (
            self.post_id, self.author_guid, self.title, self.description, self.keywords)


# could be a 'paragraph' or caption
class Target_Article_Item(Base):
    __tablename__ = 'target_article_items'

    post_id = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    author_guid = Column(Unicode, ForeignKey('posts.author_guid', ondelete="CASCADE"), primary_key=True)
    type = Column(Unicode, default=None, primary_key=True)
    item_number = Column(Integer, default=None, primary_key=True)
    content = Column(Unicode, default=None)

    def __repr__(self):
        return "<Target_Article_Item(post_id='%s', type='%s', item_number='%s', content='%s')>" % (
            self.post_id, self._author_guid, self.type, self.item_number, self.content)


class Text_From_Image(Base):
    __tablename__ = 'image_hidden_texts'

    post_id = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    author_guid = Column(Unicode, ForeignKey('posts.author_guid', ondelete="CASCADE"), primary_key=True)
    media_path = Column(Unicode, default=None)
    content = Column(Unicode, default=None)

    def __repr__(self):
        return "<Image_Hidden_Text(post_id='%s', author_guid='%s', media_path='%s', content='%s')>" % (
            self.post_id, self.author_guid, self.media_path, self.content)


class Image_Tags(Base):
    __tablename__ = 'image_tags'

    post_id = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    author_guid = Column(Unicode, ForeignKey('posts.author_guid', ondelete="CASCADE"), primary_key=True)
    media_path = Column(Unicode, default=None)
    tags = Column(Unicode, default=None)

    def __repr__(self):
        return "<Image_Tags(post_id='%s', author_guid='%s', media_path='%s', tags='%s')>" % (
            self.post_id, self.author_guid, self.media_path, self.tags)


class AuthorCitation(Base):
    __tablename__ = 'author_citations'
    # author_id_from = Column(Integer,ForeignKey("authors.author_id",ondelete="CASCADE"),primary_key=True)
    # author_id_from = Column(Integer,primary_key=True)
    from_author = Column(Unicode, primary_key=True)
    from_domain = Column(Unicode, primary_key=True)
    # author_id_to = Column(Integer,ForeignKey("authors.author_id",ondelete="CASCADE"),primary_key=True)
    # author_id_to = Column(Integer,primary_key=True)
    to_author = Column(Unicode, primary_key=True)
    to_domain = Column(Unicode, primary_key=True)
    window_start = Column(dt, primary_key=True)
    window_end = Column(dt, primary_key=True, default=None)
    number_of_citations = Column(Integer, default=None)
    from_author_guid = Column(Integer, ForeignKey("authors.author_guid", ondelete="CASCADE"))
    to_author_guid = Column(Integer, ForeignKey("authors.author_guid", ondelete="CASCADE"))

    def __repr__(self):
        return "<AuthorCitation(window_start='%s',from_author='%s',from_domain='%s',to_author='%s',to_domain='%s',number_of_citations='%s',from_author_guid='%s',to_author_guid='%s')>" % (
            self.window_start, self.from_author, self.from_domain, self.to_author, self.to_domain,
            self.number_of_citations,
            self.from_author_guid, self.to_author_guid)


class AuthorFeatures(Base):
    __tablename__ = 'author_features'
    author_guid = Column(Unicode, primary_key=True)
    window_start = Column(dt, primary_key=True)
    window_end = Column(dt, primary_key=True)
    attribute_name = Column(Unicode, primary_key=True)
    attribute_value = Column(Unicode)

    def __repr__(self):
        return "<AuthorFeatures(author_guid='%s', window_start='%s', window_end='%s', attribute_name='%s', attribute_value='%s')> " % (
            self.author_guid, self.window_start, self.window_end, self.attribute_name, self.attribute_value)

    def __init__(self, _author_guid=None, _window_start=None, _window_end=None, _attribute_name=None,
                 _attribute_value=None):
        self.author_guid = _author_guid
        self.window_start = _window_start
        self.window_end = _window_end
        self.attribute_name = _attribute_name
        self.attribute_value = _attribute_value


class Author_boost_stats(Base):
    __tablename__ = 'authors_boost_stats'

    window_start = Column(dt, default=None, primary_key=True)
    window_end = Column(dt, default=None)
    # author_id = Column(Integer,ForeignKey("authors.author_id"),primary_key=True)
    # author_id = Column(Integer,default=None) #@todo: remove field. use name and domain. reinsert author_id appropriately.
    author_name = Column(Integer, default=None, primary_key=True)
    author_domain = Column(Integer, default=None, primary_key=True)  # @todo: add domain values
    boosting_timeslots_participation_count = Column(Integer, default=None)
    count_of_authors_sharing_boosted_posts = Column(Integer, default=None)
    num_of_pointers = Column(Integer, default=None)
    num_of_pointed_posts = Column(Integer, default=None)
    pointers_scores = Column(Unicode, default=None)
    scores_sum = Column(FLOAT, default=None)
    scores_avg = Column(FLOAT, default=None)
    scores_std = Column(FLOAT, default=None)
    author_guid = Column(Unicode, default=None)

    def __repr__(self):
        return "<Author_boost_stats(window_start='%s', window_end='%s',boosting_timeslots_participation_count='%s',count_of_authors_sharing_boosted_posts='%s',num_of_pointers='%s',num_of_pointed_posts='%s',pointers_scores='%s',scores_sum='%s',scores_avg='%s',scores_std='%s',author_guid='%s')>" % (
            self.window_start, self.window_end, self.boosting_timeslots_participation_count,
            self.count_of_authors_sharing_boosted_posts, self.num_of_pointers, self.num_of_pointed_posts,
            self.pointers_scores, self.scores_sum, self.scores_avg, self.scores_std, self.author_guid)


class Post_to_pointers_scores(Base):
    __tablename__ = 'posts_to_pointers_scores'
    post_id_to = Column(Integer, ForeignKey("post_citations.post_id_to"), primary_key=True)
    window_start = Column(dt, primary_key=True)
    window_end = Column(dt, default=None)
    url_to = Column(Unicode, default=None)
    # author_id_from = Column(Integer,ForeignKey("authors.author_id"),primary_key=True)
    # author_id_from = Column(Integer,default=None)#@todo: remove field. use name and domain. reinsert author_id appropriately.
    author_name = Column(Integer, default=None, primary_key=True)
    author_domain = Column(Integer, default=None, primary_key=True)  # @todo: add domain values
    datetime = Column(Unicode, primary_key=True)
    pointer_score = Column(FLOAT, default=None)

    def __repr__(self):
        return "<Post_to_pointers_scores(post_id_to='%s', window_start='%s',window_end='%s', url_to='%s',author_id_from='%s',dt='%s',pointer_score='%s')>" % (
            self.post_id_to, self.window_start, self.window_end, self.url_to, self.author_id_from, self.dt,
            self.pointer_score)


class Posts_representativeness(Base):
    __tablename__ = 'posts_representativeness'

    post_id = Column(Unicode, ForeignKey("posts.post_id"), primary_key=True)
    topic_id = Column(Integer, primary_key=True)
    url = Column(Unicode, default=None)
    how_many_times_cited_in_topic = Column(Integer, default=None)
    in_how_many_topics = Column(Integer, default=None)
    post_count = Column(Integer, default=None)
    tfidf = Column(FLOAT, default=None)
    tof = Column(Integer, default=None)

    def __repr__(self):
        return "<Posts_representativeness(post_id='%s', topic_id='%s',url='%s', how_many_times_cited_in_topic='%s', in_how_many_topics='%s', post_count='%s', tfidf='%s', tof='%s')>" % \
               (self.post_id, self.topic_id, self.url, self.how_many_times_cited_in_topic, self.in_how_many_topics,
                self.post_count, self.tfidf, self.tof)


class AnchorAuthor(Base):
    __tablename__ = 'anchor_authors'

    author_guid = Column(Unicode, ForeignKey("authors.author_guid"), primary_key=True)
    author_type = Column(Unicode, default=None)

    def __init__(self, _author_guid, _author_type):
        self.author_guid = _author_guid
        self.author_type = _author_type

    def __repr__(self):
        return "<TestAuthors(author_guid='%s', author_type='%s')>" % \
               (self.author_guid, self.author_type)


class RandomAuthorForGraph(Base):
    __tablename__ = 'random_authors_for_graphs'

    author_guid = Column(Unicode, ForeignKey("authors.author_guid"), primary_key=True)
    author_type = Column(Unicode, default=None)

    def __repr__(self):
        return "<RandomAuthorForGraph(author_guid='%s', author_type='%s')>" % \
               (self.author_guid, self.author_type)


class SinglePostByAuthor(Base):
    __tablename__ = 'single_post_by_author'

    post_id = Column(Unicode, primary_key=True)
    author_guid = Column(Unicode, primary_key=True)
    date = Column(dt)
    content = Column(Unicode)
    domain = Column(Unicode)

    def __repr__(self):
        return "<SinglePostByAuthor(post_id='%s', author_guid='%s',date='%s', content='%s', domain='%s')>" % \
               (self.post_id, self.author_guid, self.date, self.content, self.domain)


class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)


class Post_to_topic(Base):
    __tablename__ = "posts_to_topic"

    topic_id = Column(Integer, ForeignKey("topics.topic_id"), primary_key=True)
    window_start = Column(dt, default=None, primary_key=True)
    window_end = Column(dt, default=None)
    post_id = Column(Integer, ForeignKey("posts.post_id"), primary_key=True)
    guid = Column(Unicode, default=None)
    url = Column(Unicode, default=None)

    def __repr__(self):
        return "<Post_to_topic(topic_id='%s',window_start='%s',window_end='%s', post_id='%s',guid='%s', url='%s')>" % (
            self.topic_id, self.window_start, self.window_end, self.post_id, self.guid, self.url)


class PostTopicMapping(Base):
    __tablename__ = "post_topic_mapping"

    post_id = Column(Unicode, ForeignKey("posts.post_id"), primary_key=True)
    max_topic_dist = Column(FLOAT, default=None)
    max_topic_id = Column(Integer, default=None)


class Term(Base):
    __tablename__ = "terms"

    term_id = Column(Integer, primary_key=True)
    description = Column(Unicode, default=None)

    def __repr__(self):
        return "<Term(term_id='%s',description='%s')>" % (
            self.term_id, self.description)


class Topic(Base):
    __tablename__ = "topics"

    topic_id = Column(Integer, primary_key=True)
    description = Column(Unicode, default=None)
    term_id = Column(Integer, ForeignKey("terms.term_id"), primary_key=True)
    probability = Column(FLOAT, default=None)

    def __repr__(self):
        return "<Topic(topic_id='%s',description='%s',term_id='%s', probability='%s')>" % (
            self.topic_id, self.description, self.term_id, self.probability)


class Politifact_Liar_Dataset(Base):
    __tablename__ = "politifact_liar_dataset"

    post_guid = Column(Unicode, ForeignKey("posts.guid"), primary_key=True)
    original_id = Column(Integer, default=None)
    statement = Column(Unicode, default=None)
    targeted_label = Column(Unicode, default=None)
    dataset_affiliation = Column(Unicode, default=None)
    subject = Column(Unicode, default=None)
    speaker = Column(Unicode, default=None)
    speaker_job_title = Column(Unicode, default=None)
    state_info = Column(Unicode, default=None)
    party_affiliation = Column(Unicode, default=None)
    barely_true_count = Column(Integer, default=None)
    false_count = Column(Integer, default=None)
    half_true_count = Column(Integer, default=None)
    mostly_true_count = Column(Integer, default=None)
    pants_on_fire_count = Column(Integer, default=None)
    context = Column(Unicode, default=None)

    def __repr__(self):
        return "<Politifact_Liar_Dataset(post_guid='%s', original_id='%s', statement='%s',targeted_label='%s')>" % (
            self.post_guid, self.original_id, self.statement, self.targeted_label)


class Claim_Tweet_Connection(Base):
    __tablename__ = "claim_tweet_connection"

    claim_id = Column(Unicode, primary_key=True)  # PolitiFact post
    post_id = Column(Unicode, primary_key=True)  # crawled tweet by


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column(Unicode, primary_key=True, index=True)
    title = Column(Unicode, default=None)
    description = Column(Unicode, default=None)
    url = Column(Unicode, default=None)
    verdict_date = Column(dt, default=None)
    keywords = Column(Unicode, default=None)
    domain = Column(Unicode, default=None)
    verdict = Column(Unicode, default=None)
    category = Column(Unicode, default=None)
    sub_category = Column(Unicode, default=None)

    def __repr__(self):
        return "<Claim(claim_id='%s', title='%s', description='%s', url='%s', vardict_date='%s', keywords='%s', domain='%s', verdicy='%s')>" % (
            self.claim_id, self.title, self.description, self.url, self.verdict_date, self.keywords, self.domain,
            self.verdict)


class Claim_Keywords_Connections(Base):
    __tablename__ = "claim_keywords_connections"
    claim_id = Column(Unicode, primary_key=True, index=True)
    type = Column(Unicode, primary_key=True, index=True)
    keywords = Column(Unicode, default=None)
    score = Column(FLOAT, default=None)
    tweet_count = Column(Integer, default=None)


class RedditPostCommentConnection(Base):
    __tablename__ = "reddit_post_comment_connection"
    post_id = Column(Unicode, primary_key=True, index=True)
    comment_id = Column(Unicode, primary_key=True, index=True)


class RedditPost(Base):
    __tablename__ = 'reddit_posts'
    post_id = Column(Unicode, primary_key=True, index=True)
    guid = Column(Unicode, default=None)
    link_in_body = Column(Unicode, default=None)
    ups = Column(Integer, default=0)
    downs = Column(Integer, default=0)
    score = Column(Integer, default=0)
    upvote_ratio = Column(Integer, default=0)
    number_of_comments = Column(Integer, default=None)
    parent_id = Column(Unicode, default=None)
    stickied = Column(Boolean, default=False)
    is_submitter = Column(Boolean, default=False)
    distinguished = Column(Unicode, default=None)


class RedditAuthor(Base):
    __tablename__ = 'reddit_authors'

    name = Column(Unicode, primary_key=True)
    author_guid = Column(Unicode, primary_key=True)
    comments_count = Column(Integer, default=0)
    comment_karma = Column(Integer, default=0)
    link_karma = Column(Integer, default=0)
    is_gold = Column(Boolean, default=False)
    is_moderator = Column(Boolean, default=False)
    is_employee = Column(Boolean, default=False)


class InstagramPost(Base):
    __tablename__ = 'instagram_posts'
    id = Column(Unicode, primary_key=True)
    display_url = Column(Unicode, default=None)
    comments_disabled = Column(Boolean, default=None)
    likes = Column(Integer, default=None)
    # body = Column(Unicode, default=None)
    comment_count = Column(Integer, default=None)
    is_video = Column(Boolean, default=None)
    # owner_id = Column(Unicode, default=None)
    shortcode = Column(Unicode, default=None)
    # taken_at_timestamp = Column(Integer, default=None)
    thumbnail_resources = Column(Unicode, default=None)
    media_preview = Column(Unicode, default=None)
    gating_info = Column(Unicode, default=None)
    dimensions = Column(Unicode, default=None)
    instagram_typename = Column(Unicode, default=None)
    hashtag = Column(Unicode, default=None)


class InstagramAuthor(Base):
    __tablename__ = 'instagram_authors'
    id = Column(Unicode, primary_key=True)
    # username = Column(Unicode, default=None)
    # full_name = Column(Unicode, default=None)
    # biography = Column(Unicode, default=None)
    followers_count = Column(Integer, default=None)
    following_count = Column(Integer, default=None)
    posts_count = Column(Integer, default=None)
    is_business_account = Column(Boolean, default=None)
    is_joined_recently = Column(Boolean, default=None)
    is_private = Column(Boolean, default=None)
    # profile_pic_url = Column(Unicode, default=None)


class GooglePostKeywords(Base):
    __tablename__ = 'google_post_keywords'

    post_id = Column(Integer, primary_key=True)
    keywords = Column(Unicode, primary_key=True)
    insertion_date = Column(Unicode, default=None)


class NewsArticle(Base):
    __tablename__ = 'news_articles'

    article_id = Column(Unicode, ForeignKey('claims.claim_id', ondelete="CASCADE"), primary_key=True)
    author = Column(Unicode, default=None)
    published_date = Column(dt, default=None)
    domain = Column(Unicode, default=None)
    url = Column(Unicode, default=None)
    title = Column(Unicode, default=None)
    description = Column(Unicode, default=None)
    content = Column(Unicode, default=None)
    url_to_image = Column(Unicode, default=None)

    def __repr__(self):
        return "<TargetArticle(post_id='%s', author_guid='%s', author='%s', published_date='%s', url='%s', title='%s', description='%s', keywords='%s')>" % (
            self.post_id, self.author_guid, self.author, self.published_date, self.url, self.title, self.description,
            self.keywords)


class News_Article_Item(Base):
    __tablename__ = 'news_article_items'

    post_id = Column(Unicode, ForeignKey('posts.post_id', ondelete="CASCADE"), primary_key=True)
    author_guid = Column(Unicode, ForeignKey('posts.author_guid', ondelete="CASCADE"), primary_key=True)
    source_newsapi_internal_id = Column(Unicode, default=None)
    source_newsapi_internal_name = Column(Unicode, default=None)
    content = Column(Unicode, default=None)
    img_url = Column(Unicode, default=None)

    def __repr__(self):
        return "<Target_Article_Item(post_id='%s', author_guid='%s', source_newsapi_internal_id='%s', source_newsapi_internal_name='%s', content='%s', img_url='%s')>" % (
            self.post_id, self._author_guid, self.source_newsapi_internal_id, self.source_newsapi_internal_name,
            self.content, self.img_url)


class UkraineRussiaConflictPOI(Base):
    __tablename__ = 'ukraine_russia_conflict_pois'

    user_screen_name = Column(Unicode, primary_key=True)
    poi_type = Column(Unicode, default=None)
    url = Column(Unicode, default=None)
    original_screen_name = Column(Unicode, default=None)

    def __repr__(self):
        return "<POI(user_screen_name='%s', poi_type='%s', url='%s')>" % (self.user_screen_name, self.poi_type, self.url)


class TwitterUserIdFollowersNextCursor(Base):
    __tablename__ = 'twitter_user_id_followers_next_cursor'

    user_id = Column(Unicode, primary_key=True)
    cursor = Column(Unicode, default=None)

    def __repr__(self):
        return "<TwitterUserIdFollowersNextCursor(user_id='%s', cursor='%s', url='%s')>" % (self.user_id, self.cursor)



class DB():
    '''
    Represents the primary blackboard of the system.
    The module must be the first one to setUp.
    '''

    def __init__(self):
        pass

    def setUp(self):
        configInst = getConfig()
        self._date = getConfig().eval(self.__class__.__name__, "start_date")
        self._pathToEngine = configInst.get(self.__class__.__name__, "DB_path") + \
                             configInst.get(self.__class__.__name__, "DB_name_prefix") + \
                             configInst.get("DEFAULT", "social_network_name") + \
                             configInst.get(self.__class__.__name__, "DB_name_suffix")

        start_date = configInst.get("DEFAULT", "start_date").strip("date('')")
        self._window_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self._window_size = datetime.timedelta(
            seconds=int(configInst.get("DEFAULT", "window_analyze_size_in_sec")))
        self._window_end = self._window_start + self._window_size

        if configInst.eval(self.__class__.__name__, "remove_on_setup"):
            self.deleteDB()

        self.engine = create_engine("sqlite:///" + self._pathToEngine, echo=False)
        self.Session = sessionmaker()
        self.Session.configure(bind=self.engine)

        self.session = self.Session()

        self.posts = "posts"
        self.authors = "authors"
        self.author_features = "author_features"

        @event.listens_for(self.engine, "connect")
        def connect(dbapi_connection, connection_rec):
            dbapi_connection.enable_load_extension(True)
            if (getConfig().eval("OperatingSystem", "windows")):
                full_path = os.path.abspath("%s%s") % (configInst.get("DB", "DB_path_to_extension"), '.dll')
                dbapi_connection.execute('SELECT load_extension("{}")'.format(full_path).replace('\\', '/'))
            if (getConfig().eval("OperatingSystem", "linux")):
                dbapi_connection.execute(
                    'SELECT load_extension("%s%s")' % (configInst.get("DB", "DB_path_to_extension"), '.so'))
            if (getConfig().eval("OperatingSystem", "mac")):
                dbapi_connection.execute(
                    'SELECT load_extension("%s%s")' % (configInst.get("DB", "DB_path_to_extension"), '.dylib'))

            dbapi_connection.enable_load_extension(False)

        if getConfig().eval(self.__class__.__name__, "dropall_on_setup"):
            Base.metadata.drop_all(self.engine)

        Base.metadata.create_all(self.engine)
        pass

    def tearDown(self):
        if getConfig().eval(self.__class__.__name__, "dropall_on_teardown"):
            if (os.path.exists(self._pathToEngine)):
                Base.metadata.drop_all(self.engine)

        if getConfig().eval(self.__class__.__name__, "remove_on_teardown"):
            self.deleteDB()

        if getConfig().eval(self.__class__.__name__, "vacuum_db"):
            self.vacuum_db()

    def vacuum_db(self):
        query = text("VACUUM;")
        self.session.execute(query)

    def execute(self, window_start):
        pass

    def cleanUp(self, window_start):
        pass

    def canProceedNext(self, window_start):
        return True

    def is_well_defined(self):
        return True

    ##########################################################
    # miscellaneous
    def deleteDB(self):
        if (os.path.exists(self._pathToEngine)):
            try:
                os.remove(self._pathToEngine)
            except:
                logging.exception("Data Base %s remove failed" % self._pathToEngine)

    def commit(self):
        self.session.commit()

    def is_post_topic_mapping_table_exist(self):
        query = text("SELECT name FROM sqlite_master WHERE type='table' AND name='post_topic_mapping'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return len(records) != 0

    def is_topics_table_exist(self):
        query = text("SELECT name FROM sqlite_master WHERE type='table' AND name='topics'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return len(records) != 0

    def is_table_exist(self, table_name):
        q = "SELECT name FROM sqlite_master WHERE type='table' AND name=" + "\'" + table_name + "\'"
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return len(records) != 0

    def get_key_posts(self):
        query = text("SELECT post_id FROM export_key_posts")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return [rec[0] for rec in records]

    def delete_post_representativeness_data(self):
        query = text("DELETE FROM posts_representativeness;")
        self.session.execute(query)

    def get_retweets_with_no_tweet_citation(self):
        '''
        :return: a list of post_ids and urls of retweets whose connection doesn't contain any reference to twitter
        '''
        query = text("select posts.post_id as post_id_from, posts.url as url_from " \
                     "from posts " \
                     "where posts.content like \'%RT @%\' " \
                     "Except "
                     "select post_citations.post_id_from, post_citations.url_from from post_citations where post_citations.url_to like \'%twitter.com%\'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return {rec[0]: rec[1] for rec in records}

    def is_post_citation_exist(self, post_id_from, post_id_to):
        query = text(
            "select * from post_citations where post_citations.post_id_from = :post_id_from and post_citations.post_id_to = :post_id_to")
        result = self.session.execute(query, params=dict(post_id_from=post_id_from, post_id_to=post_id_to))
        cursor = result.cursor
        records = list(cursor.fetchall())
        return len(records) > 0

    def get_topic_to_author_mapping(self, target_author_field):
        '''
        :return: a mapping of <topic_id> -> <author name> -> <number of posts in the topic> for each topic
        '''
        ans = {}
        query = text("""select max_topic_id, authors.{0}, count(*) as posts_in_topic_count 
                         from post_topic_mapping , posts  , authors 
                         where post_topic_mapping.post_id = posts.post_id 
                         and authors.author_guid = posts.author_guid 
                         group by max_topic_id, author 
                         order by max_topic_id""".format(target_author_field))
        result = self.session.execute(query)
        for topic_id, author, posts_in_topic_count in result:
            if not topic_id in ans:
                ans[topic_id] = {}
            ans[topic_id][author] = posts_in_topic_count
        return ans

    def get_topics(self):
        query = text("select * from topics")
        result = self.session.execute(query)
        return [r for r in result]

    def update_json_post_retweeter(self, id, key, value):
        update_query = "UPDATE " + self.post_retweeter_table + " SET " + key + "=" + str(
            value) + " WHERE retweeter_id=" + str(id)
        self.update_query(update_query)

    def update_query(self, query):
        self.session.execute(query)
        self.session.commit()

    def get_json_post_retweeter(self, post_id, retweeter_id):
        query = "SELECT * FROM " + self.post_retweeter_table + \
                " WHERE post_id=" + str(post_id) + " AND retweeter_id=" + str(retweeter_id)
        result = self.session.execute(query)
        cursor = result.cursor
        post_retweeter_result = cursor.fetchall()

        if len(post_retweeter_result):
            twitter_user = self.create_post_retweeter(post_retweeter_result)
            return twitter_user
        return None

    def encode_field_into_utf8(self, text):
        if text is not None:
            return str(text)
        return text

    ###########################################################
    # posts
    ###########################################################

    def create_object(self, query_result):

        object = query_result[0]

        '''
        post.post_id = values[0]
        post.author_id = values[1]
        post.post_twitter_id = values[2]
        post.post_vico_guid = values[3]
        post.text = values[4]
        post.title = values[5]
        post.retweet_count = values[6]
        post.favorites_count = values[7]
        post.created_at = values[8]
        post.url = values[9]
        post.is_detailed = values[10]
        post.is_LB = values[11]
        post.domain = values[12]
        '''
        return object

    def delete_post(self, post_id):
        # delete_query = "DELETE FROM " + self.posts + " WHERE post_id=" + str(post_id)
        # self.session.execute(delete_query)
        # self.session.commit()

        self.session.query(Post).filter(Post.post_id == post_id).delete()
        self.session.commit()

    def get_post_by_id(self, post_id):

        query = self.session.query(Post).filter(Post.post_id == post_id)
        posts_result = query.all()

        # query = "SELECT * FROM " + self.posts + " WHERE post_id=" + str(post_id)
        # result = self.session.execute(query)
        # cursor = result.cursor
        # posts_result = cursor.fetchall()

        if len(posts_result):
            post = self.create_object(posts_result)
            return post
        return None

    def get_posts(self):
        entries = self.session.query(Post).all()
        return entries

    def get_posts_without_retweet_connections(self):
        entries = self.session.query(Post).filter(and_(~exists().where(Post.post_id==PostRetweeterConnection.post_osn_id), Post.domain != 'retweet')).all()
        return entries

    def get_posts_with_retweet_connections(self):
        entries = self.session.query(Post).filter(
            and_(exists().where(Post.post_id == PostRetweeterConnection.post_osn_id), Post.domain != 'retweet')).all()
        return entries

    def get_posts_retweets_dict_gen(self, limit = 10):
        source_posts = aliased(Post, name='source_posts')
        retweets = aliased(Post, name='retweets')
        # entries = self.session.query(source_posts, retweets).filter(
        #     and_(source_posts.post_id == PostRetweeterConnection.post_osn_id,
        #          retweets.post_id == PostRetweeterConnection.retweeter_twitter_id,
        #          source_posts.domain != u'retweet')).yield_per(10000).enable_eagerloads(False)
        entries = self.session.query(PostRetweeterConnection.post_osn_id, PostRetweeterConnection.retweeter_twitter_id)
        source_retweets_dict = defaultdict(list)
        for source_id, retweet_id in entries:
            source_retweets_dict[source_id].append(retweet_id)
            if len(list(source_retweets_dict.keys())) == limit:
                yield source_retweets_dict
                source_retweets_dict = defaultdict(list)
        yield source_retweets_dict

    def get_author_connection_by_source(self, source_ids):
        entries = self.session.query(AuthorConnection.source_author_guid, AuthorConnection.destination_author_guid).filter(AuthorConnection.source_author_guid.in_(set(source_ids)))
        author_connection_dict = defaultdict(set)
        for source_id, dest_id in entries:
            author_connection_dict[source_id].add(dest_id)
        return author_connection_dict

    def get_claims(self):
        return self.session.query(Claim).all()

    def get_claims_without_tweets(self):
        claims_with_tweets = self.session.query(Claim_Tweet_Connection.claim_id)
        return self.session.query(Claim).filter(~Claim.claim_id.in_(claims_with_tweets)).all()

    def get_claims_without_keywords(self):
        claims_with_keywords = self.session.query(Claim_Keywords_Connections.claim_id)
        return self.session.query(Claim).filter(~Claim.claim_id.in_(claims_with_keywords)).all()

    def get_claims_by_domain(self, domain):
        return self.session.query(Claim).filter(Claim.domain == domain).all()

    def get_posts_with_no_dates(self):
        records = self.session.query(Post).filter(Post.date == '2007-01-01 00:00:00').all()
        return records

    def get_all_posts(self):
        entries = self.session.query(Post).all()
        return entries

    def get_post_dictionary(self):
        posts = self.session.query(Post).yield_per(100000).enable_eagerloads(False)
        post_id_post_dict = defaultdict()
        for i, post in enumerate(posts):
            print('\r load posts {}'.format(str(i+1)), end="")
            post_id = post.post_id
            post_id_post_dict[post_id] = post
        print()
        return post_id_post_dict

    def get_elements_by_args(self, args, offset=0, author_guids=None):
        source_table = args['source']['table_name']
        source_id = args['source']['id']
        source_where_clauses = []
        if 'where_clauses' in args['source']:
            source_where_clauses = args['source']['where_clauses']

        connection_table_name = args['connection']['table_name']
        connection_source_id = args['connection']['source_id']
        connection_targeted_id = args['connection']['target_id']
        connection_where_clauses = []
        if 'where_clauses' in args['connection']:
            connection_where_clauses = args['connection']['where_clauses']

        destination_table_name = args['destination']['table_name']
        destination_id = args['destination']['id']
        destination_where_clauses = []
        if 'where_clauses' in args['destination']:
            destination_where_clauses = args['destination']['where_clauses']

        source_table = aliased(self.get_table_by_name(source_table), name="source")
        connection_table = self.get_table_by_name(connection_table_name)
        destination_table = aliased(self.get_table_by_name(destination_table_name), name="dest")
        connection_conditions = self._get_connection_conditions(connection_where_clauses, destination_table,
                                                                source_table)

        source_conditions = self._get_conditions_from_where_cluases(source_table, source_where_clauses)
        destination_conditions = self._get_conditions_from_where_cluases(destination_table, destination_where_clauses)
        conditions = source_conditions + destination_conditions + connection_conditions
        source_id_attr = getattr(source_table, source_id)
        connection_source_attr = getattr(connection_table, connection_source_id)
        connection_target_attr = getattr(connection_table, connection_targeted_id)
        destination_id_attr = getattr(destination_table, destination_id)

        if author_guids:
            conditions.append(~connection_source_attr.in_(set(author_guids)))

        table_elements = self.session.query(connection_source_attr, destination_table) \
            .join(source_table, connection_source_attr == source_id_attr) \
            .join(destination_table, connection_target_attr == destination_id_attr) \
            .filter(and_(condition for condition in conditions)) \
            .order_by(connection_source_attr) \
            .yield_per(10000).enable_eagerloads(False).offset(offset)

        return table_elements

    def _get_connection_conditions(self, connection_where_clauses, destination_table, source_table):
        connection_conditions = []
        for where_clause in connection_where_clauses:
            val1 = where_clause["val1"]
            val2 = where_clause["val2"]
            val1_attr = self._get_table_attr_by_prefix(destination_table, source_table, val1)
            val2_attr = self._get_table_attr_by_prefix(destination_table, source_table, val2)
            op_name = where_clause["op"]
            if op_name == "timeinterval":
                delta = where_clause["delta"]
                binary_exp1 = func.datetime(val1_attr, "-{0} day".format(delta)) <= val2_attr
                binary_exp2 = func.datetime(val1_attr, "+{0} day".format(delta)) >= val2_attr
                connection_conditions.append(binary_exp1)
                connection_conditions.append(binary_exp2)

            elif op_name == "before":
                delta = where_clause["delta"]
                binary_exp1 = func.datetime(val1_attr, "-{0} day".format(delta)) <= val2_attr
                binary_exp2 = val1_attr > val2_attr
                connection_conditions.append(binary_exp1)
                connection_conditions.append(binary_exp2)
            elif op_name == "after":
                delta = where_clause["delta"]
                binary_exp1 = val1_attr <= val2_attr
                binary_exp2 = func.datetime(val1_attr, "+{0} day".format(delta)) >= val2_attr
                connection_conditions.append(binary_exp1)
                connection_conditions.append(binary_exp2)
            else:

                binary_exp = val1_attr.op(op_name)(val2_attr)
                connection_conditions.append(binary_exp)
        return connection_conditions

    def _get_table_attr_by_prefix(self, destination_table, source_table, val1):
        if "source" in val1:
            val1_attr = getattr(source_table, val1.replace('source.', ''))
        elif "dest" in val1:
            val1_attr = getattr(destination_table, val1.replace('dest.', ''))
        else:
            val1_attr = val1
        return val1_attr

    def get_table_elements_by_ids(self, table_name, id_field, ids, where_cluases=[]):
        table_elements = self.get_table_elements_by_where_cluases(table_name, where_cluases, id_field)
        ids_set = set(ids)
        table_elements = [element for element in table_elements if getattr(element, id_field) in ids_set]
        return table_elements

    def get_table_elements_by_where_cluases(self, table_name, where_cluases, data_id, offset=0, author_guids=None):
        table = self.get_table_by_name(table_name)
        conditions = self._get_conditions_from_where_cluases(table, where_cluases)
        if author_guids:
            conditions.append(~getattr(table, data_id).in_(set(author_guids)))

        table_elements = self.session.query(table) \
            .filter(and_(condition for condition in conditions)) \
            .order_by(data_id) \
            .yield_per(10000).enable_eagerloads(False).offset(offset)
        return table_elements

    def _get_conditions_from_where_cluases(self, table, where_cluases):
        conditions = []
        for where_clause_dict in where_cluases:
            field_name = where_clause_dict['field_name']
            value = where_clause_dict['value']
            op_name = '='
            if 'op' in where_clause_dict:
                op_name = where_clause_dict['op']
            try:
                table_attr = getattr(table, field_name)
                binary_exp = table_attr.op(op_name)(value)
            except:
                table_attr = field_name
                binary_exp = field_name == value
            conditions.append(binary_exp)
        return conditions

    def get_table_dictionary(self, table_name, table_id):
        table = self.get_table_by_name(table_name)
        posts = self.session.query(table).all()
        post_id_post_dict = defaultdict()
        for post in posts:
            post_id = getattr(post, table_id)
            post_id_post_dict[post_id] = post
        return post_id_post_dict

    def get_filterd_author_dict(self, author_ids):
        authors = self.session.query(Author).filter(Author.author_guid.in_(author_ids)).yield_per(10000).enable_eagerloads(False)
        author_id_author_dict = defaultdict()
        for i, author in enumerate(authors):
            print('\rload authors {}'.format(i), end='')
            author_guid = getattr(author, 'author_guid')
            author_id_author_dict[author_guid] = author
        print()
        return author_id_author_dict

    def get_filterd_source_dict(self, source_ids, table_name, key_name):
        table = self.get_table_by_name(table_name)
        key_field = getattr(table, key_name)

        items = self.session.query(table).filter(key_field.in_(source_ids)).yield_per(10000).enable_eagerloads(False)
        source_id_source_dict = defaultdict()
        for i, item in enumerate(items):
            print('\rload {} {}'.format(table_name, str(i+1)), end='')
            item_key = getattr(item, key_name)
            source_id_source_dict[item_key] = item
        print()
        return source_id_source_dict

    def get_author_dictionary(self):
        return self.authors_dict_by_field('author_guid')

    def authors_dict_by_field(self, field='author_guid'):
        authors = self.session.query(Author).yield_per(10000).enable_eagerloads(False)
        author_id_author_dict = defaultdict()
        for i, author in enumerate(authors):
            print('\rload authors {}'.format(i), end='')
            author_guid = getattr(author, field)
            author_id_author_dict[author_guid] = author
        print()
        return author_id_author_dict

    def get_author_guid_posts_dict(self):
        posts = self.session.query(Post).yield_per(10000).enable_eagerloads(False)
        author_guid_posts_dict = defaultdict(list)
        for post in posts:
            author_guid_posts_dict[post.author_guid].append(post)
        return author_guid_posts_dict

    def _get_posts_content_screen_name_tuples(self):
        logging.info("Get all post content")
        q = "SELECT DISTINCT posts.content,posts.author " \
            "FROM posts " \
            "WHERE posts.domain = 'Microblog'"

        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        return cursor
        # records = list(cursor.fetchall())
        # return records

    def get_posts_by_domain(self, domain):
        posts_by_user = {}
        # posts = self.session.query(Post).filter(Post.domain == unicode(domain)).slice(start,stop).all()
        query = text("select posts.author_guid, posts.date, posts.content from posts where posts.domain = :domain "
                     "and length(posts.content)>0 and posts.date IS NOT NULL")
        counter = 0
        print("schema_definition.get_posts_by_domain before executing query..")
        result = self.session.execute(query, params=dict(domain=domain))
        print("schema_definition.get_posts_by_domain finished executing query..")
        cursor = result.cursor
        print("schema_definition.get_posts_by_domain before calling generator function")
        posts = self.result_iter(cursor, arraysize=10000)
        print("schema_definition.get_posts_by_domain after calling generator function")

        posts_by_user = self._create_user_posts_dictinary(posts)
        # TODO added by lior, needs to verify that it doesn't break anything
        if len(posts_by_user) == 0:
            guid = self.get_author_guids()
            posts_by_user = {author: () for author in guid}
        return posts_by_user

    def get_author_posts_dict_by_minimal_num_of_posts(self, domain, min_num_of_posts):
        query = """
                SELECT posts.author_guid, posts.date, posts.content
                FROM posts
                WHERE posts.author_guid IN (
	              SELECT posts.author_guid
	              FROM posts
	              WHERE posts.domain = :domain
	              AND LENGTH(posts.content)>0
	              GROUP BY posts.author_guid
	              HAVING COUNT(*) >= :min_num_of_posts
                )
                """
        # posts = self.session.query(Post).filter(Post.domain == unicode(domain)).slice(start,stop).all()
        query = text(query)
        result = self.session.execute(query, params=dict(domain=domain, min_num_of_posts=min_num_of_posts))
        cursor = result.cursor
        posts = self.result_iter(cursor, arraysize=10000)
        posts_by_user = self._create_user_posts_dictinary(posts)
        return posts_by_user

    def get_random_author_posts_dict_by_minimal_num_of_posts(self):
        query = """
                SELECT posts.author_guid, posts.date, posts.content
                FROM posts
                WHERE LENGTH(posts.content)>0
                AND posts.author_guid IN (
                  SELECT random_authors_for_graphs.author_guid
                  FROM random_authors_for_graphs
                )
                """
        # posts = self.session.query(Post).filter(Post.domain == unicode(domain)).slice(start,stop).all()
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        posts = self.result_iter(cursor)  # , arraysize=10000)
        posts_by_user = self._create_user_posts_dictinary(posts)
        return posts_by_user

    def _create_user_posts_dictinary(self, posts):
        posts_by_user = defaultdict(list)
        counter = 0
        for current_post in posts:
            counter += 1
            if counter % 5000 == 0:
                msg = "\r Creating post objects " + str(counter)
                print(msg, end="")
            str_date = current_post[1]
            date_obj = datetime.datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S')
            post = Struct(author_guid=current_post[0], date=date_obj, content=current_post[2])

            posts_by_user[str(post.author_guid)].append(post)
        return posts_by_user

    def get_single_post_per_author_topic_mapping(self):
        q = " select * from single_post_per_author_topic_mapping"
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        result = list(cursor.fetchall())
        return result

    def get_posts_by_author_guid(self, author_guid):

        query = self.session.query(Post).filter(Post.author_guid == author_guid).order_by(Post.date)
        entries = query.all()
        return entries

    """
    if window_start and window_end is not given search in all DB
    """

    def isPostExist(self, url, window_start=None, window_end=None):

        if window_start is None or window_end is None:
            query = text("SELECT EXISTS(SELECT * FROM posts WHERE (url= :url)  limit 1)")
            result = self.session.execute(query, params=dict(url=str(url)))
            return [r for (r,) in result][0]
        else:
            query = text(
                "SELECT EXISTS(SELECT * FROM posts WHERE (url= :url or guid= :guid) and (:window_start <= date and "
                "date <=:window_end) limit 1)")
            result = self.session.execute(query,
                                          params=dict(url=str(url), window_start=window_start, window_end=window_end))
            return [r for (r,) in result][0]

    def addPost(self, post):
        self.session.merge(post)

    def addPosts(self, posts):
        logging.info("total Posts inserted to DB: " + str(len(posts)))
        i = 1
        self.session.flush()
        for post in posts:
            if (i % 100 == 0):
                msg = "\r Insert post to DB: [{}".format(i) + "/" + str(len(posts)) + ']'
                print(msg, end="")
            i += 1
            self.addPost(post)
        # self.session.flush()
        self.session.commit()
        if len(posts) != 0: print("")

    def merge_items(self, items):
        logging.info("total items inserted to DB: " + str(len(items)))
        for i, item in enumerate(items):
            if (i % 100 == 0):
                msg = "\r Insert items to DB: [{}".format(str(i + 1)) + "/" + str(len(items)) + ']'
                print(msg, end="")
            self.session.merge(item)

    def updatePost(self, post):
        self.session.query(Post).filter(Post.url == post[0]).update(post[1])

    def updatePosts(self, posts):
        logging.info("total Posts updated to DB: " + str(len(posts)))
        i = 1
        for post in posts:
            msg = "\r update post to DB: [{}".format(i) + "/" + str(len(posts)) + ']'
            print(msg, end="")
            i += 1
            self.updatePost(post)
        self.session.commit()
        if len(posts) != 0: print("")

    def getPostUsingURL(self, url, window_start=None, window_end=None):
        if window_start is None or window_end is None:
            query = self.session.query(Post).filter(Post.url == url)
        else:
            query = self.session.query(Post).filter(
                and_(Post.url == url, window_start <= Post.date, Post.date <= window_end))
        return query.all()

    def isRefExist(self, url):
        q = text("SELECT EXISTS(SELECT * FROM posts WHERE url= :url limit 1)")
        res = self.session.execute(q, params=dict(url=str(url)))
        return [r for (r,) in res][0]

    def isPostNotDetailed(self, url, guid):
        q = text("SELECT EXISTS(SELECT * FROM posts WHERE (url= :url or guid= :guid) and \
            is_detailed=0 limit 1)")
        res = self.session.execute(q, params=dict(url=str(url), guid=str(guid)))
        return [r for (r,) in res][0]

    def addReference(self, reference):
        self.session.merge(reference)

    def addReferences(self, references):
        i = 1
        for ref in references:
            msg = "\r Add ref: [{}".format(i) + "/" + str(len(references)) + ']'
            print(msg, end="")
            i += 1
            self.addReference(ref)
        self.session.commit()

    def getPostsMaxDate(self, window_start=None, window_end=None):
        if window_start is None or window_end is None:
            res = self.session.query(func.max(Post.date))
        else:
            res = self.session.query(func.max(Post.date)).filter(
                and_(Post.date >= window_start, Post.date <= window_end))
        return res.scalar()

    def contains_post(self, post_url):
        q = text("select * from posts where posts.url = :post_url")
        res = self.session.execute(q, params=dict(post_url=post_url))
        res = [r for r in res]
        return len(res) > 0

    ###########################################################
    # authors
    ###########################################################

    def insertIntoAuthorsTable(self, win_start, win_end):
        # TODO: remove window_start and window_end
        q = text(
            "insert or ignore into authors(name,domain,author_guid, xml_importer_insertion_date) select distinct author,domain,author_guid, xml_importer_insertion_date from posts where author_guid>''")
        self.session.execute(q)
        self.session.commit()

    def insert_or_update_authors_from_xml_importer(self, win_start, win_end):
        authors_to_update = []
        posts = self.session.query(Post).filter(Post.author_guid != "").all()
        logging.info("Insert or update_authors from xml importer")
        logging.info("total Posts: " + str(len(posts)))
        i = 1
        for post in posts:
            msg = "\r Insert or update posts: [{}".format(i) + "/" + str(len(posts)) + ']'
            print(msg, end="")
            i += 1
            author_guid = post.author_guid
            domain = post.domain
            result = self.get_author_by_author_guid_and_domain(author_guid, domain)
            if not result:
                author = Author()
                author.name = post.author
                author.domain = post.domain
                author.author_guid = post.author_guid
            else:
                author = result[0]
            author.xml_importer_insertion_date = post.xml_importer_insertion_date
            authors_to_update.append(author)
        if len(posts) != 0: print("")
        self.add_authors(authors_to_update)

    def addAuthor(self, author):
        self.session.merge(author)

    def addAuthors(self, authorsList):
        logging.info("total Posts inserted to DB: " + str(len(authorsList)))
        i = 1
        for author in authorsList:
            if (i % 100 == 0):
                msg = "\r Insert author to DB: [{}".format(i) + "/" + str(len(authorsList)) + ']'
                print(msg, end="")
            i += 1
            self.addAuthor(author)
        self.commit()

    def insert_authors(self):
        query = text(
            "insert or ignore into authors(author_screen_name) select distinct author_screen_name from posts where author_screen_name>''")
        self.session.execute(query)
        self.session.commit()

    def get_authors(self):
        result = self.session.query(Author).all()
        return result

    def get_authors_media_pats(self):
        result = self.session.query(Author.media_path).filter(Author.media_path.isnot(None)).all()
        return result

    def get_authors_withot_connection(self, connection_type):
        # query = self.session.query(Author).filter(and_(Author.author_guid.in_(AuthorConnection.source_author_guid),
        #                                                AuthorConnection.connection_type == connection_type))

        query = text("""SELECT *
                        FROM authors
                        WHERE authors.author_guid  NOT IN (SELECT source_author_guid 
                                                           FROM author_connections
                                                           WHERE connection_type = :connection_type)
                     """)
        result = self.session.execute(query, params=dict(connection_type=connection_type))
        author_dicts = list(map(dict, result))
        authors = [Author(**author_dict) for author_dict in author_dicts]
        return authors

    def get_authors_with_connections(self, connection_type):
        # query = self.session.query(Author).filter(and_(Author.author_guid.in_(AuthorConnection.source_author_guid),
        #                                                AuthorConnection.connection_type == connection_type))

        query = text("""SELECT source_author_guid 
                       FROM author_connections
                       WHERE connection_type = :connection_type
                     """)
        result = self.session.execute(query, params=dict(connection_type=connection_type))
        cursor = result.cursor
        tuples = cursor.fetchall()
        authors = set(chain(*tuples))
        return authors

    def get_reddit_authors(self):
        result = self.session.query(RedditAuthor).all()
        return result

    def get_reddit_posts(self):
        result = self.session.query(RedditPost).all()
        return result

    def get_all_authors(self):
        result = self.session.query(Author).all()
        return result

    def get_authors_by_domain(self, domain):
        targeted_social_network = getConfig().get("DEFAULT", "social_network_name")
        # if targeted_social_network == Social_Networks.TWITTER:
        #     result = self.session.query(Author).filter(and_(Author.domain == unicode(domain)),
        #                                                 Author.author_osn_id.isnot(None),
        #                                                 or_(Author.xml_importer_insertion_date.isnot(None), Author.mark_missing_bad_actor_retweeters_insertion_date.isnot(None))).all()
        # else:
        result = self.session.query(Author).filter(and_(Author.domain == str(domain))
                                                   ).all()

        return result

    def get_temp_author_connections_all(self):
        result = self.session.query(TempAuthorConnection).all()

        return result

    def get_authors_by_domain_dict(self):
        authors = self.get_authors()
        author_domain_dict = defaultdict(list)
        for author in authors:
            author_domain_dict[author.domain].append(author)
        return author_domain_dict

    def get_author_guid_to_author_dict(self):
        authors = self.get_all_authors()
        authors_dict = dict((aut.author_guid, aut) for aut in authors)
        return authors_dict

    # def get_authors_by_domain(self, domain):
    #     targeted_social_network = getConfig().get("DEFAULT", "social_network_name")
    #     if targeted_social_network == Social_Networks.TWITTER:
    #         result = self.session.query(Author).filter(and_(Author.domain == unicode(domain)),
    #                                                     Author.author_osn_id.isnot(None),
    #                                                     or_(Author.xml_importer_insertion_date.isnot(None), Author.mark_missing_bad_actor_retweeters_insertion_date.isnot(None))).all()
    #     else:
    #         result = self.session.query(Author).filter(and_(Author.domain == unicode(domain)),
    #                                                    Author.author_osn_id.isnot(None)
    #                                                    ).all()
    #
    #     return result

    # def get_authors(self, domain):
    #     result = self.session.query(Author).filter(and_(Author.domain == unicode(domain),
    #                                                     Author.author_osn_id.isnot(None))
    #                                                ).all()
    #
    #     return result

    def get_number_of_targeted_osn_authors(self, domain):
        query = text("""SELECT COUNT(authors.author_guid)
                        FROM authors
                        WHERE authors.domain = :domain
                        AND authors.author_osn_id IS NOT NULL
                        AND (authors.xml_importer_insertion_date IS NOT NULL
                        OR authors.mark_missing_bad_actor_retweeters_insertion_date IS NOT NULL)""")
        result = self.session.execute(query, params=dict(domain=domain))
        cursor = result.cursor
        tuples = cursor.fetchall()
        if tuples is not None and len(tuples) > 0:
            authors_count = tuples[0][0]
            return authors_count
        return None

    def get_number_of_authors(self):
        query = text("""SELECT COUNT(authors.author_guid)
                        FROM authors""")
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = cursor.fetchall()
        if tuples is not None and len(tuples) > 0:
            authors_count = tuples[0][0]
            return authors_count
        return None

    def get_number_of_targeted_osn_posts(self, domain):
        query = text("""SELECT COUNT(posts.author_guid)
                        FROM posts
                        WHERE posts.domain = :domain""")
        result = self.session.execute(query, params=dict(domain=domain))
        cursor = result.cursor
        tuples = cursor.fetchall()
        if tuples is not None and len(tuples) > 0:
            posts_count = tuples[0][0]
            return posts_count
        return None

    def get_number_of_posts(self):
        query = text("""SELECT COUNT(posts.author_guid)
                        FROM posts""")
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = cursor.fetchall()
        if tuples is not None and len(tuples) > 0:
            posts_count = tuples[0][0]
            return posts_count
        return None

    def get_author_by_guid(self, guid):
        ans = self.session.query(Author).filter(Author.author_guid == guid).all()
        return ans[0]

    def getAuthorByName(self, name):
        logging.info("Name of given author is: " + name)
        return self.session.query(Author).filter(Author.name == name).all()

    def getAuthorIDbyNameAndDomain(self, name, start_wind, domain):
        res = self.session.query(Author.author_guid).filter(Author.name == name and Author.domain == domain).all()

        return [r for (r,) in res][0]

    def get_author_guid_and_author_osn_id(self, domain):
        data = {}
        query = text(" SELECT author_guid, author_osn_id \
                      FROM authors \
                      WHERE(authors.xml_importer_insertion_date IS NOT NULL \
                              OR authors.mark_missing_bad_actor_retweeters_insertion_date IS NOT NULL ) \
                        AND authors.author_osn_id IS NOT NULL AND authors.domain = :domain ")
        res = self.session.execute(query, params=dict(domain=domain))
        all_rows = res.cursor.fetchall()

        for row in all_rows:
            data[row[0]] = row[1]
        return data

    '''
    def get_author_by_id(self, author_id):

        query = self.session.query(Author).filter(Author.author_id == author_id)
        posts_result = query.all()

        #query = "SELECT * FROM " + self.posts + " WHERE post_id=" + str(post_id)
        #result = self.session.execute(query)
        #cursor = result.cursor
        #posts_result = cursor.fetchall()

        if len(posts_result):
            post = self.create_object(posts_result)
            return post
        return None
    '''

    def delete_author(self, name, domain, author_guid):
        self.session.query(Author).filter(
            (Author.name == name) & (Author.domain == domain) & (Author.author_guid == author_guid)).delete()
        self.session.commit()

    def update_author(self, author):
        self.session.merge(author)

    def update_author_type_by_author_guid(self, guid, type):
        self.session.query(Author).filter(Author.author_guid == guid).update({'author_type': type})
        self.session.commit()

    def get_author_name_by_post_content(self, post_content):
        query = text("select posts.author from posts where posts.content like :post_content")
        res = self.session.execute(query, params=dict(post_content=post_content + "%"))
        return [author_name[0] for author_name in res]

    ###########################################################
    # author_citations
    ###########################################################

    def deleteAuthCit(self, window_start=None):
        if window_start:
            self.session.query(AuthorCitation).filter(AuthorCitation.window_start == window_start).delete()

        else:
            self.session.query(AuthorCitation).delete()
        self.session.commit()
        pass

    def insertIntoAuthorCitation(self, win_start, win_end):

        q = text(
            " insert into author_citations (from_author, from_domain, to_author, to_domain, window_start, window_end, number_of_citations,from_author_guid,to_author_guid) \
            select \
                p1.author as from_author, \
                p1.domain as from_domain, \
                p2.author as to_author, \
                p2.domain as to_domain,  \
                :window_start as window_start, \
                :window_end as window_end, \
                count(*) as number_of_citations, \
                p1.author_guid as from_author_guid, \
                p2.author_guid as to_author_guid \
            from         \
              post_citations as ref \
              inner join posts as p1 on p1.post_id=ref.post_id_from \
              inner join posts as p2 on p2.post_id=ref.post_id_to \
            where \
                p2.author_guid>'' and p1.author_guid>'' and \
                :window_start <= p1.date and p1.date <= :window_end \
            group by from_author, from_domain, to_author, to_domain ")

        self.session.execute(q, params=dict(window_start=win_start, window_end=win_end))
        self.commit()

        ###########################################################
        # author_features
        ###########################################################

    def get_author_features_by_author_guid(self, author_guid):
        result = self.session.query(AuthorFeatures).filter(AuthorFeatures.author_guid == author_guid).all()
        if len(result) > 0:
            return result
        return None

    def get_author_feature(self, author_guid, attribute_name):
        result = self.session.query(AuthorFeatures).filter(and_(AuthorFeatures.author_guid == author_guid,
                                                                AuthorFeatures.attribute_name == attribute_name)).all()
        if len(result) > 0:
            return result[0]
        return None

    def get_author_features(self):

        result = self.session.query(AuthorFeatures).all()
        # if len(result) > 0:
        return result
        # return None

    def get_author_features_labeled_authors_only(self):
        query = text('select author_features.*  \
                from  \
                  author_features  \
                inner join authors on (author_features.author_guid = authors.author_guid)  \
                where authors.author_type is not null')
        result = self.session.execute(query)
        cursor = result.cursor
        author_features = cursor.fetchall()
        return author_features

    def get_author_features_lazy(self, offset=0, yield_per=10000):
        result = self.session.query(AuthorFeatures).yield_per(yield_per).enable_eagerloads(False).offset(offset)
        return result

    def get_author_features_labeled_authors_only_lazy(self, offset=0, yield_per=10000):
        result = self.session.query(AuthorFeatures)\
            .join(source_table, AuthorFeatures.author_guid == Author.author_guid)\
            .filter(authors.author_type.isnot(None))\
            .yield_per(yield_per).enable_eagerloads(False).offset(offset)
        return result


    # def get_author_features_by_author_id_field(self, author_id_field, target_field, is_labeled, ):
    #     logging.info("Start getting authors features")
    #     if is_labeled:
    #         query = text('select author_features.*, authors.'+target_field+' \
    #                 from  \
    #                   author_features  \
    #                 inner join authors on (author_features.author_guid = authors.'+author_id_field+')  \
    #                 where authors.author_type is not null')
    #     #     query = text('select author_features.*, authors.'+target_field+' \
    #     #             from  \
    #     #               author_features  \
    #     #             inner join authors on (author_features.author_guid = authors.name) where authors.author_type is not null')
    #     else:
    #         query = text('select author_features.*, authors.' + target_field + ' \
    #                              from  \
    #                                author_features  \
    #                              inner join authors on (author_features.author_guid = authors.' + author_id_field + ')  \
    #                              where authors.author_type is null')
    #     result = self.session.execute(query)
    #     cursor = result.cursor
    #     author_features = cursor.fetchall()
    #     logging.info("Finished getting authors features")
    #
    #     return author_features

    def get_author_features_by_author_id_field(self, author_id_field, target_field, is_labeled):
        logging.info("Start getting authors features")
        if is_labeled:
            query = text(
                'select author_features.*, authors.' + target_field + ' from author_features inner join authors on (author_features.author_guid = authors.' + author_id_field + ') where authors.author_type is not null')
        #     query = text('select author_features.*, authors.'+target_field+' \
        #             from  \
        #               author_features  \
        #             inner join authors on (author_features.author_guid = authors.name) where authors.author_type is not null')
        else:
            query = text('select author_features.*, authors.' + target_field + ' \
                                 from  \
                                   author_features  \
                                 inner join authors on (author_features.author_guid = authors.' + author_id_field + ')  \
                                 where authors.author_type is null')
        result = self.session.execute(query)
        cursor = result.cursor
        author_features = cursor.fetchall()
        logging.info("Finished getting authors features")

        return author_features

    def get_author_features_unlabled_authors_only_by_author_id_field(self, author_id_field, target_field):
        logging.info("Start getting authors features")

        query = text('select author_features.*, authors.' + target_field + ' \
                        from  \
                          author_features  \
                        inner join authors on (author_features.author_guid = authors.' + author_id_field + ')  \
                        where authors.author_type is null')
        result = self.session.execute(query)
        cursor = result.cursor
        author_features = cursor.fetchall()
        logging.info("Finished getting authors features")
        return author_features

    def insert_authors_features(self, list_author_features):
        self.session.add_all(list_author_features)

    def update_author_features(self, author_features):
        self.session.merge(author_features)

    def update_target_articles(self, target_article):
        self.session.merge(target_article)

    def update_image_hidden_text(self, image_hidden_text):
        self.session.merge(image_hidden_text)

    def add_author_features(self, author_features):
        logging.info("total Author Features inserted to DB: " + str(len(author_features)))
        i = 1
        for author_feature in author_features:
            if (i % 100 == 0):
                msg = "\r Insert author featurs to DB: [{}".format(i) + "/" + str(len(author_features)) + ']'
                print(msg, end="")
            i += 1
            self.update_author_features(author_feature)
        self.commit()

    def convert_auhtor_feature_author_id_form_author_name_to_author_guid(self):
        query = "update author_features SET author_guid = (select author_guid from authors where authors.name = author_features.author_guid)"
        self.session.execute(q)
        self.commit()

    def add_target_articles(self, target_articles):
        logging.info("target_articles inserted to DB: " + str(len(target_articles)))
        i = 1
        for target_article in target_articles:
            if (i % 100 == 0):
                msg = "\r Insert target_article to DB: [{}".format(i) + "/" + str(len(target_articles)) + ']'
                print(msg, end="")
            i += 1
            self.update_target_articles(target_article)
        self.commit()

    def add_image_hidden_texts(self, image_hidden_texts):
        logging.info("image_hidden_texts inserted to DB: " + str(len(image_hidden_texts)))
        i = 1
        for image_hidden_text in image_hidden_texts:
            if (i % 100 == 0):
                msg = "\r Insert image_hidden_text to DB: [{}".format(i) + "/" + str(len(image_hidden_texts)) + ']'
                print(msg, end="")
            i += 1
            self.update_image_hidden_text(image_hidden_text)
        self.commit()

    def delete_authors_features(self):
        q = text("delete from author_features")
        self.session.execute(q)
        self.commit()

    def delete_from_authors_features_trained_authors(self, author_guids_to_remove):
        self.session.query(AuthorFeatures).filter(AuthorFeatures.author_guid.in_(author_guids_to_remove)).delete(
            synchronize_session='fetch')
        self.session.commit()

    ###########################################################
    # key_authors
    ###########################################################
    def get_key_authors(self):
        query = text("SELECT author_name FROM export_key_authors")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return [rec[0] for rec in records]

    def get_sum_tfidf_scores(self):
        '''
        :return: A map of author_guid->sumtfidf
        '''
        query = text("SELECT export_key_authors.author_guid, "
                     "export_key_authors.SumTFIDF "
                     "FROM export_key_authors "
                     "JOIN authors "
                     "ON export_key_authors.author_guid = authors.author_guid "
                     "WHERE domain='Microblog'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return {rec[0]: rec[1] for rec in records}

    def get_max_tfidf_scores(self):
        '''
        :return: A map author_guid->maxtfidf
        '''
        query = text("SELECT export_key_authors.author_guid, "
                     "export_key_authors.MaxTFIDF "
                     "FROM export_key_authors "
                     "JOIN authors "
                     "ON export_key_authors.author_guid = authors.author_guid "
                     "WHERE domain='Microblog'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return {rec[0]: rec[1] for rec in records}

    def is_export_key_authors_view_exist(self):
        query = text("SELECT name FROM sqlite_master WHERE type='view' AND name='export_key_authors'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        return len(records) != 0

    ###########################################################
    # author_boost_scores
    ###########################################################
    def deleteBoostAuth(self, window_start=None):
        if window_start:
            self.session.query(Post_to_pointers_scores).filter(
                Post_to_pointers_scores.window_start == window_start).delete()
            self.session.query(Author_boost_stats).filter(Author_boost_stats.window_start == window_start).delete()

        else:
            self.session.query(Post_to_pointers_scores).delete()
            self.session.query(Author_boost_stats).delete()
        self.session.commit()
        pass

    def getPostsListWithoutEmptyRowsByDate(self, window_start, window_end):

        q = text("select * from posts where content is not NULL and (:window_start <= date and date <= :window_end)")
        references = []
        res = self.session.execute(q, params=dict(window_start=window_start, window_end=window_end))
        posts = [list(post.values()) for post in res]
        return posts

    def getPostsListWithoutEmptyRowsByDomain(self, domain):

        q = text("select * from posts where content is not NULL and domain = :domain")
        references = []
        res = self.session.execute(q, params=dict(domain=domain))
        posts = [list(post.values()) for post in res]
        return posts

    def getPostsListWithoutEmptyRows(self):
        q = text("select * from posts where content is not NULL")
        references = []
        res = self.session.execute(q)
        posts = [list(post.values()) for post in res]
        return posts

    def get_author_guid_post_dict(self):
        author_guid_posts_dict = defaultdict(list)
        posts = self.get_posts()
        for post in posts:
            author_guid_posts_dict[post.author_guid].append(post)
        return author_guid_posts_dict

    def addAuthor_boost_stats(self, author_boost_stats):
        self.session.merge(author_boost_stats)

    def addAuthors_boost_stats(self, authors_boost_stats):
        for author_boost_stats in authors_boost_stats:
            self.addAuthor_boost_stats(author_boost_stats)
        self.session.commit()

    def addPost_to_pointers_scores(self, post_to_pointers_scores):
        self.session.merge(post_to_pointers_scores)

    def addPosts_to_pointers_scores(self, posts_to_pointers_scores):
        for post_to_pointers_scores in posts_to_pointers_scores:
            self.addPost_to_pointers_scores(post_to_pointers_scores)
        self.session.commit()

    def getReferencesFromPost(self, postid):
        urlToList = []
        references = self.session.query(Post_citation).filter(Post_citation.post_id_from == postid).all()
        for ref in references:
            urlToList.append(ref.url_to)
        return list(set(urlToList))

    def getAuthor_boost_stats(self, author_guid):
        result = self.session.query(Author_boost_stats).filter(Author_boost_stats.author_guid == author_guid).all()

        if len(result) > 0:
            return result[0]
        return None

    def get_all_authors_boost_stats(self):
        result = self.session.query(Author_boost_stats).filter(
            Author_boost_stats.author_domain == str(domain)).all()

        if len(result) > 0:
            return result
        return None

    ###########################################################
    # post_retweeter_connections
    ###########################################################
    def get_post_retweeter_connections_by_post_id(self, post_id):
        return self.session.query(PostRetweeterConnection).filter(PostRetweeterConnection.post_osn_id == post_id).all()

    ###########################################################
    # authors
    ###########################################################

    def add_author(self, author):
        self.session.merge(author)

    def add_authors(self, authors):
        logging.info("-- add_authors --")
        logging.info("Number of authors is: " + str(len(authors)))
        i = 1
        for author in authors:
            msg = "\r Add author to DB: [{}".format(i) + "/" + str(len(authors)) + ']'
            print(msg, end="")
            i += 1
            self.add_author(author)
        self.commit()
        if len(authors) != 0: print("")

    def get_author_by_author_guid(self, author_guid):
        result = self.session.query(Author).filter(Author.author_guid == author_guid).all()
        return result[0]

    def get_author_by_screen_name(self, screen_name):
        result = self.session.query(Author).filter(Author.author_screen_name == screen_name).all()
        return result[0]

    def get_author_by_author_guid_and_domain(self, author_guid, domain):
        result = self.session.query(Author).filter(and_(Author.author_guid == author_guid,
                                                        Author.domain == domain)).all()
        return result

    def is_author_exists(self, author_guid, domain):
        author = self.get_author_by_author_guid_and_domain(author_guid, domain)
        return len(author) > 0

    def get_authors_retrieved_from_xml_importer(self):
        result = self.session.query(Author).filter(Author.xml_importer_insertion_date is not None).all()
        return result

    def get_retweet_count(self):
        query = text("""select author_guid, count(posts.post_id)
                        from posts
                        where content like '%RT @%'
                        group by author_guid""")
        result = self.session.execute(query)
        records = list(result.cursor.fetchall())
        return {rec[0]: rec[1] for rec in records}

    def get_retweets(self):
        query = text(" select post_id, content from posts where content like '%RT @%'")
        result = self.session.execute(query)
        records = list(result.cursor.fetchall())
        return {rec[0]: rec[1] for rec in records}

    def get_missing_data_twitter_screen_names(self):
        """
        This function retrieves all the users who have missing information.
        These users are authors who has no twitter user id and their posts with url of twitter
        :return: list of screen names
        """
        query = "SELECT authors.name " \
                "FROM authors " \
                "WHERE authors.author_osn_id is NULL " \
            # "AND (authors.is_suspended_or_not_exists is NULL " \
        # "OR authors.is_suspended_or_not_exists = 0) " \
        # "AND authors.domain = 'Microblog' " \
        # "AND authors.author_type IS NULL  " \
        # "GROUP BY authors.name" #TODO: Domains.MICROBLOG
        # "LIMIT 35;"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = list(cursor.fetchall())
        twitter_screen_names = [r[0] for r in screen_names]
        return twitter_screen_names

    def get_missing_data_twitter_screen_names_by_posts(self):
        query = "SELECT DISTINCT(posts.author) " \
                "FROM posts WHERE LOWER(posts.author) NOT IN " \
                "(SELECT LOWER(author_screen_name) FROM authors WHERE author_osn_id IS NOT NULL)"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = list(cursor.fetchall())
        twitter_screen_names = [r[0] for r in screen_names]
        return twitter_screen_names

    def get_missing_data_twitter_screen_names_by_authors(self):
        query = "SELECT DISTINCT(author_screen_name) " \
                "FROM authors WHERE author_osn_id IS NULL"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = list(cursor.fetchall())
        twitter_screen_names = [r[0] for r in screen_names]
        return twitter_screen_names

    def get_authors_for_mark_as_suspended_or_not_existed(self):
        # This function retrieves all the users who are suspended or not existing.
        # We should run this query only when we finished to retrieve all the followers
        # after saving all of them we can be sure that twitter did not bring them due to their situation(suspended)
        query = "SELECT * " \
                "FROM authors " \
                "INNER JOIN posts on (authors.author_guid = posts.author_guid) " \
                "WHERE authors.author_osn_id is NULL " \
                "AND authors.name = posts.author " \
                "AND (posts.url LIKE  \'%http://twitter.com%\' " \
                "OR posts.url LIKE \'%https://twitter.com%\')"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        suspended_authors = list(cursor.fetchall())
        return suspended_authors

    def get_not_suspended_authors(self, domain):
        result = self.session.query(Author).filter(
            and_(Author.is_suspended_or_not_exists == None, Author.domain == domain)).all()
        return result

    def get_followers_brought_by_terms(self):
        # query = """
        #             SELECT DISTINCT(authors.author_osn_id)
        #             FROM author_connections
        #             INNER JOIN authors ON (authors.author_guid = author_connections.destination_author_guid)
        #             WHERE author_connections.connection_type = 'term-author'
        #             AND authors.author_osn_id NOT IN (
        #                 SELECT DISTINCT(temp_author_connections.source_author_osn_id)
        #                 FROM temp_author_connections
        #                 WHERE temp_author_connections.connection_type = 'follower'
        #             )
        #         """
        # AND authors.author_osn_id IS NOT '18691328'
        query = """
                    SELECT DISTINCT(authors.author_osn_id)
                    FROM author_connections
                    INNER JOIN authors ON (authors.author_guid = author_connections.destination_author_guid)
                    WHERE author_connections.connection_type = 'term-author'

                    AND authors.author_osn_id NOT IN (
                        SELECT DISTINCT(temp_author_connections.source_author_osn_id)
                        FROM temp_author_connections
                        WHERE temp_author_connections.connection_type = 'follower'
                    )
                """

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = cursor.fetchall()
        author_osn_ids = [tuple[0] for tuple in tuples]
        return author_osn_ids

    def get_followers_or_friends_candidats(self, connection_type, domain, limit):
        # This function retrieves all the user ids we would like to extract their followers.
        # These users are authors who are not the source in the connections table and not in follower type.
        # Moreover, bring me the authors who are not protected, and they are from twitter (type = microblog)

        # query = """
        #        SELECT *
        #         FROM (SELECT authors.author_guid
        #         FROM authors
        #         WHERE authors.author_guid NOT IN(
        #         SELECT source_author_guid
        #         FROM  author_connections
        #         WHERE connection_type = :connection_type)
        #         AND authors.protected = 0
        #         AND authors.author_type is NULL
        #         AND authors.domain = :domain
        #         AND authors.xml_importer_insertion_date IS NOT NULL
        #         AND authors.followers_count > 10
        #         LIMIT 5)
        # 	Union
        #         SELECT  author_guid
        #         FROM (
        #         SELECT authors.author_guid,   authors.followers_count,  authors.statuses_count
        #         FROM authors
        #         WHERE authors.author_type = 'bad_actor'
        #         AND authors.statuses_count > 10
        #         AND authors.followers_count > 10
        #         AND authors.protected = 0
        #         AND authors.author_guid NOT IN (
        #         SELECT author_connections.source_author_guid
        #         FROM author_connections
        #         WHERE author_connections.connection_type = :connection_type)
        #         ORDER BY authors.followers_count DESC, authors.statuses_count DESC
        #         LIMIT 0)
        #         """

        query = """
                SELECT authors.author_osn_id
                FROM authors
                WHERE authors.author_guid NOT IN(
                                                                        SELECT source_author_guid
                                                                        FROM  author_connections
                                                                        WHERE connection_type = :connection_type)
                AND authors.protected = 0
                AND authors.domain = :domain
                AND {0} > 10
                LIMIT :limit
        """
        if (connection_type == Author_Connection_Type.FOLLOWER):
            query = query.format('authors.followers_count')
        elif (connection_type == Author_Connection_Type.FRIEND):
            query = query.format('authors.friends_count')
        else:
            query = query.format('11')
        query = text(query)
        result = self.session.execute(query, params=dict(connection_type=connection_type, domain=domain, limit=limit))
        cursor = result.cursor
        return cursor

    def get_twitter_author_ids_for_extracting_friends(self):
        # This function retrieves all the user ids we would like to extract their followers.
        # These users are authors who are not the source in the connections table and not in follower type.
        # Moreover, bring me the authors who are not protected, and they are from twitter (type = microblog)
        '''
        query = "SELECT authors.author_osn_id " \
                "FROM authors " \
                "WHERE authors.author_osn_id NOT IN " \
                "(SELECT authors.author_osn_id " \
                "FROM authors " \
                "INNER JOIN author_connections ON " \
                "(authors.author_osn_id = author_connections.source_author_osn_id) " \
                "WHERE author_connections.connection_type = 'friend') " \
                "AND authors.protected = 0 " \
                "AND authors.domain = 'Microblog' " \
                "AND authors.xml_importer_insertion_date IS NOT NULL  " \
                "AND authors.missing_data_complementor_insertion_date IS NOT NULL " \
                "AND authors.friends_count > 0 " \
                #"LIMIT 20;"
        '''
        query = "SELECT * " \
                "FROM (" \
                "SELECT authors.author_osn_id " \
                "FROM authors " \
                "WHERE authors.author_osn_id NOT IN(" \
                "SELECT author_connections.source_author_osn_id " \
                "FROM  author_connections " \
                "WHERE author_connections.connection_type = 'friend') " \
                "AND authors.protected = 0 " \
                "AND authors.author_type is NULL " \
                "AND authors.domain = 'Microblog' " \
                "AND authors.xml_importer_insertion_date IS NOT NULL " \
                "AND authors.friends_count > 10 " \
                "LIMIT 35) " \
                "Union " \
                "SELECT  author_osn_id " \
                "FROM (" \
                "SELECT authors.author_osn_id,   authors.friends_count,  authors.statuses_count " \
                "FROM authors " \
                "WHERE authors.author_type = 'bad_actor' " \
                "AND authors.statuses_count > 10 " \
                "AND authors.friends_count > 10 " \
                "AND authors.protected = 0 " \
                "AND authors.author_osn_id NOT IN (" \
                "SELECT author_connections.source_author_osn_id " \
                "FROM author_connections " \
                "WHERE author_connections.connection_type = 'friend') " \
                "ORDER BY authors.friends_count DESC, authors.statuses_count DESC " \
                "LIMIT 35)"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        ids = list(cursor.fetchall())
        twitter_ids = [r[0] for r in ids]
        return twitter_ids

    def get_screen_names_for_twitter_authors_by_posts(self):
        screen_names = []
        http_twitter_prefix = str('%http://twitter.com%')
        https_twitter_prefix = str('%https://twitter.com%')
        results = self.session.query(Post.url).filter(and_(Post.xml_importer_insertion_date is not None,
                                                           Post.author == Author.name,
                                                           Author.missing_data_complementor_insertion_date is None,
                                                           or_(Post.url.like(http_twitter_prefix),
                                                               Post.url.like(https_twitter_prefix)))).all()

        expression_one = r"(?<=http:\/\/twitter\.com\/)\w+(?=\/statuses\/\d+)"
        expression_two = r"(?<=https:\/\/twitter\.com\/)\w+(?=\/statuses\/\d+)"
        r_one = re.compile(expression_one, re.VERBOSE)
        r_two = re.compile(expression_two, re.VERBOSE)
        for result in results:
            twitter_url = result[0]
            optional_screen_names = r_one.findall(twitter_url)
            if optional_screen_names:
                screen_name = optional_screen_names[0]
                screen_names.append(screen_name)
            else:
                optional_screen_names = r_two.findall(twitter_url)
                if optional_screen_names:
                    screen_name = optional_screen_names[0]
                    screen_names.append(screen_name)
        return screen_names

    def get_twitter_authors_retrieved_from_vico_importer(self):
        '''
        query = text("SELECT * FROM authors JOIN posts on posts.author = authors.name WHERE posts.xml_importer_insertion_date IS NOT NULL AND authors.bad_actors_markup_insertion_date is NULL AND (posts.url LIKE '%http://twitter.com%' OR posts.url LIKE '%https://twitter.com%');")
        result = self.session.execute(query)
        authors = [author.values() for author in result]
        return authors
        '''

        result = self.session.query(Author).filter(and_(Post.xml_importer_insertion_date is not None),
                                                   (Post.author == Author.name),
                                                   (Author.missing_data_complementor_insertion_date is None),
                                                   or_(Post.url.like('%http://twitter.com%'),
                                                       Post.url.like('%https://twitter.com%'))).all()
        return result

    def add_author_connection(self, author_connection):
        self.session.merge(author_connection)

    def add_post_retweeter_connection(self, post_retweeter_connection):
        self.session.merge(post_retweeter_connection)

    def add_author_connections(self, author_connections):
        total = len(author_connections)
        current = 0
        for author_connection in author_connections:
            current += 1
            msg = '\r adding ' + str(current) + ' of ' + str(total) + ' author_connections'
            print(msg, end="")
            self.add_author_connection(author_connection)
        self.session.commit()

    def get_author_connections_dict(self):
        authors_connections = self.session.query(AuthorConnection).all();
        authors_connections_dict = defaultdict(list)
        for authors_connection in authors_connections:
            authors_connections_dict[authors_connection.connection_type].append(authors_connection)
        return authors_connections_dict

    def get_author_connections_by_author_guid(self, source_author_guid):
        return self.session.query(AuthorConnection).filter(
            AuthorConnection.source_author_guid == source_author_guid).all();

    def add_post_retweeter_connections(self, post_retweeter_connections):
        for post_retweeter_connection in post_retweeter_connections:
            self.add_post_retweeter_connection(post_retweeter_connection)
        self.session.commit()

    def get_vico_importer_bad_actors(self):
        vico_importer_bad_actors = self.session.query(Author).filter(and_(Author.author_type == Author_Type.BAD_ACTOR,
                                                                          or_(
                                                                              Author.xml_importer_insertion_date is not None,
                                                                              Author.vico_dump_insertion_date is not None))).all()
        number_of_vico_importer_bad_actors = len(vico_importer_bad_actors)
        logging.info("Number of bad_actors which found by VICO is: " + str(number_of_vico_importer_bad_actors))
        return vico_importer_bad_actors

    def get_bad_actor_ids(self):
        logging.info("get_bad_actor_retweeters_not_retrieved_from_vico")

        query = "SELECT authors.author_osn_id " \
                "FROM authors " \
                "WHERE authors.author_type = 'bad_actor'"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        ids = list(cursor.fetchall())
        twitter_authors_ids = [r[0] for r in ids]
        logging.info("Number of bad actor o is: " + str(len(twitter_authors_ids)))
        return twitter_authors_ids

        number_of_bad_actors = len(bad_actors)
        logging.info("Number of bad_actors which found by VICO is: " + str(number_of_bad_actors))
        return bad_actors

    def get_vico_importer_potential_good_actors(self):
        vico_importer_potential_good_actors = self.session.query(Author).filter(and_(Author.author_type is None,
                                                                                     Author.missing_data_complementor_insertion_date is not None,
                                                                                     Author.xml_importer_insertion_date is not None)).all()
        # or_(Author.xml_importer_insertion_date != None,
        # Author.vico_dump_insertion_date != None))).all()
        number_of_vico_importer_potential_good_actors = len(vico_importer_potential_good_actors)
        logging.info(
            "Number of vico_importer_potential_good_actors is: " + str(number_of_vico_importer_potential_good_actors))
        return vico_importer_potential_good_actors

    def convert_twitter_users_to_authors(self, users, targeted_social_network, author_type, inseration_type):
        authors = []
        seen_authors = set()
        logging.info("Convert twitter users to authors: " + str(len(users)))
        i = 1
        for user in users:
            author = self.convert_twitter_user_to_author(user, targeted_social_network, author_type, inseration_type)
            if author.author_guid not in seen_authors:
                authors.append(author)
                seen_authors.add(author.author_guid)
            msg = "\r Author record was converted: {0} [{1}/{2}]".format(author.author_screen_name, i, str(len(users)))
            print(msg, end="")
            # print("Author record was converted: " + author.author_screen_name)
            i += 1
            # logging.info("Author record was converted: " + author.author_screen_name)
        return authors

    def convert_twitter_user_to_author(self, osn_user, targeted_social_network, author_type, inseration_type):
        author_screen_name = str(osn_user.screen_name)
        author_guid = compute_author_guid_by_author_name(author_screen_name)

        domain = Domains.MICROBLOG
        # result = self.get_author_by_author_guid(author_guid)
        # if len(result) == 0:
        author = Author()
        # else:
        #     author = result[0]

        author.author_screen_name = str(author_screen_name)
        author.name = author_screen_name  # .lower()
        author.domain = str(targeted_social_network)

        author.author_guid = str(author_guid)

        author.author_full_name = str(osn_user.name)
        author.author_osn_id = str(osn_user.id)
        author.description = str(osn_user.description)
        author.created_at = str(osn_user.created_at)
        author.statuses_count = osn_user.statuses_count
        author.followers_count = osn_user.followers_count
        author.friends_count = osn_user.friends_count
        author.favourites_count = osn_user.favourites_count
        author.listed_count = osn_user.listed_count
        author.language = str(osn_user.lang)
        author.profile_background_color = osn_user.profile_background_color
        author.profile_background_tile = osn_user.profile_background_tile
        author.profile_banner_url = osn_user.profile_banner_url
        author.profile_image_url = osn_user.profile_image_url
        author.profile_link_color = osn_user.profile_link_color
        author.profile_sidebar_fill_color = osn_user.profile_sidebar_fill_color
        author.profile_text_color = osn_user.profile_text_color
        author.default_profile = osn_user.default_profile
        author.contributors_enabled = osn_user.contributors_enabled
        author.default_profile_image = osn_user.default_profile_image
        author.geo_enabled = osn_user.geo_enabled
        author.protected = osn_user.protected
        author.location = str(osn_user.location)
        author.notifications = osn_user.notifications
        author.time_zone = str(osn_user.time_zone)
        author.url = str(osn_user.url)
        author.utc_offset = osn_user.utc_offset
        author.verified = osn_user.verified
        author.is_suspended_or_not_exists = None

        if author_type is Author_Type.BAD_ACTOR:
            author.author_type = author_type
        self.set_inseration_date(author, inseration_type)

        return author

    # set date to authors
    def set_inseration_date(self, author, inseration_type):
        # now = unicode(get_current_time_as_string())
        now = self._date
        if inseration_type == DB_Insertion_Type.BAD_ACTORS_COLLECTOR:
            author.bad_actors_collector_insertion_date = now
        elif inseration_type == DB_Insertion_Type.XML_IMPORTER:
            author.xml_importer_insertion_date = now
        elif inseration_type == DB_Insertion_Type.MISSING_DATA_COMPLEMENTOR:
            author.missing_data_complementor_insertion_date = now
        elif inseration_type == DB_Insertion_Type.BAD_ACTORS_MARKUP:
            author.bad_actors_markup_insertion_date = now
        elif inseration_type == DB_Insertion_Type.MARK_MISSING_BAD_ACTOR_RETWEETERS:
            author.mark_missing_bad_actor_retweeters_insertion_date = now

    def create_author_connections(self, source_id, destination_author_ids, weight, author_connection_type,
                                  insertion_date):
        print("---create_author_connections---")
        author_connections = []
        for destination_author_id in destination_author_ids:
            author_connection = self.create_author_connection(source_id, destination_author_id, weight,
                                                              author_connection_type, insertion_date)
            author_connections.append(author_connection)

        return author_connections

    def create_author_connection(self, source_author_guid, destination_author_guid, weight, connection_type,
                                 insertion_date):
        # print("---create_author_connection---")
        author_connection = AuthorConnection()

        # msg = '\r Author connection: source -> ' + str(source_author_guid) + ', dest -> ' + str(
        #     destination_author_guid) + ', connection type = ' + connection_type
        # print(msg, end="")

        # print("Author connection: source -> " + str(source_author_guid) + ", dest -> " + str(destination_author_guid) + ", connection type = " + connection_type)
        author_connection.source_author_guid = source_author_guid
        author_connection.destination_author_guid = destination_author_guid
        author_connection.connection_type = str(connection_type)
        author_connection.weight = str(weight)
        author_connection.insertion_date = insertion_date

        return author_connection

    def save_author_connections(self, author_connections):
        print("---Saving author connections in DB---")
        save_author_connections_start_time = time.time()
        # self.add_author_connections(author_connections)
        self.add_author_connections_fast(author_connections)
        save_author_connections_end_time = time.time()
        save_author_connections_time = save_author_connections_end_time - save_author_connections_start_time
        print("Saving author connections in DB took in seconds: " + str(save_author_connections_time))

    def create_and_save_author_connections(self, source_author_id, follower_ids, weight, connection_type):
        author_connections = self.create_author_connections(source_author_id, follower_ids, weight, connection_type,
                                                            self._window_start)
        self.save_author_connections(author_connections)

    def get_author_connections_by_type(self, connection_type):
        # print("get_author_connections_by_type: " + str(connection_type))
        query = self.session.query(AuthorConnection).filter(AuthorConnection.connection_type == connection_type)
        res = self.session.execute(query)
        cursor = res.cursor
        return cursor

    def result_iter(self, cursor, arraysize=1000):
        'An iterator that uses fetchmany to keep memory usage down'
        while True:
            results = cursor.fetchmany(arraysize)
            if not results:
                break
            for result in results:
                yield result

    def get_post_min_date(self):
        query = "SELECT MIN(posts.date) " \
                "FROM posts"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        fetched_curser = cursor.fetchall()
        str_date_object = fetched_curser[0]
        str_date = str_date_object[0]
        returned_date = self.create_date_from_full_string_date(str_date)
        return returned_date

    def get_post_max_date(self):
        query = "SELECT MAX(posts.date) " \
                "FROM posts"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        fetched_curser = cursor.fetchall()
        str_date_object = fetched_curser[0]
        str_date = str_date_object[0]
        returned_date = self.create_date_from_full_string_date(str_date)
        return returned_date

    def create_date_from_full_string_date(self, str_date):
        date_and_hour = str_date.split(" ")
        str_selected_date = date_and_hour[0]
        year_month_day = str_selected_date.split("-")
        year = int(year_month_day[0])
        month = int(year_month_day[1])
        day = int(year_month_day[2])
        from datetime import date, timedelta as td
        created_date = date(year, month, day)
        return created_date

    def get_new_author_screen_names_by_date(self, min_date, current_date):

        yesterday = current_date - timedelta(days=1)
        query = "SELECT DISTINCT posts.author " \
                "FROM posts " \
                "WHERE (posts.url LIKE '%http://twitter.com%' " \
                "OR posts.url LIKE '%https://twitter.com%') " \
                "AND posts.domain = 'Microblog' " \
                "AND posts.date > " + "'" + str(current_date) + " 00:00:00' " \
                                                                "AND posts.date <= " + "'" + str(
            current_date) + " 23:59:59' " \
                            "and posts.author not in ( " \
                            "SELECT DISTINCT posts.author " \
                            "FROM posts " \
                            "WHERE (posts.url LIKE '%http://twitter.com%' " \
                            "OR posts.url LIKE '%https://twitter.com%') " \
                            "AND posts.domain = 'Microblog' " \
                            "AND posts.date > " + "'" + (
                    str(yesterday) if current_date == min_date else str(min_date)) + " 00:00:00' " \
                                                                                     "AND posts.date <= " + "'" + str(
            yesterday) + " 23:59:59' " \
                         "GROUP BY author)"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        authors = list(cursor.fetchall())
        author_screen_names = [r[0] for r in authors]
        return author_screen_names

    ###########################################################
    # Views creation
    ###########################################################
    def create_uf_view(self):
        '''
        Represents the #references to a post from a given topic
        '''
        self.session.execute("DROP VIEW IF EXISTS uf;")
        self.session.execute("\
        CREATE VIEW IF NOT EXISTS uf AS\
        SELECT p.post_id_to AS post_id, \
            p.url_to,\
            t.max_topic_id AS topic_id,\
            count(p.post_id_from) AS url_frequency \
        FROM post_citations p \
            INNER JOIN \
            post_topic_mapping t ON (p.post_id_from = t.post_id) \
        GROUP BY p.post_id_to, \
            t.max_topic_id\
            ;")
        self.session.commit()

    def create_tf_view(self):
        '''
        Represents the #topics pointing to a post
        '''
        self.session.execute("DROP VIEW IF EXISTS tf;")
        self.session.execute("\
        CREATE VIEW IF NOT EXISTS tf AS\
        SELECT uf.post_id,\
            count(uf.topic_id) AS topic_frequency \
        FROM uf\
        GROUP BY uf.post_id\
        ;")
        self.session.commit()

    def create_total_url_frequency_view(self):
        '''
        Represents #references to a post overall
        '''
        self.session.execute("DROP VIEW IF EXISTS total_url_frequency;")
        self.session.execute("\
            CREATE VIEW IF NOT EXISTS total_url_frequency AS\
            SELECT uf.post_id,\
                   sum(uf.url_frequency) AS tof\
            FROM uf \
            GROUP BY uf.post_id;\
            ")
        self.session.commit()

    def create_topic_stats_view(self):
        '''
        Represents how many references is available from each topic
        '''
        self.session.execute("DROP VIEW IF EXISTS topic_stats;")
        self.session.execute("\
            CREATE VIEW IF NOT EXISTS topic_stats as \
            SELECT max_topic_id AS topic_id, count(post_id) as post_count \
            FROM post_topic_mapping \
            GROUP BY max_topic_id \
            ")
        self.session.commit()

    def create_tfidf_view(self):
        self.session.execute("DROP VIEW IF EXISTS tfidf;")
        self.session.execute("\
         CREATE VIEW IF NOT EXISTS tfidf AS \
             SELECT uf.post_id,\
                    uf.topic_id,\
                    uf.url_to,\
                    uf.url_frequency AS how_many_times_cited_in_topic,\
                    tf.topic_frequency AS in_how_many_topics,\
                    topic_stats.post_count,\
                    1.0 * uf.url_frequency / topic_stats.post_count * log(1.0 *  (select count(*) from topics) / tf.topic_frequency) AS ufitf1,\
                    tuf.tof\
               FROM uf\
                    INNER JOIN\
                    tf ON (uf.post_id = tf.post_id) \
                    INNER JOIN \
                    topic_stats ON (uf.topic_id = topic_stats.topic_id) \
                    INNER JOIN\
                    total_url_frequency tuf ON (tuf.post_id = uf.post_id);\
         ")
        self.session.commit()

    def create_key_posts_view(self):
        self.session.execute("DROP VIEW IF EXISTS export_key_posts;")
        self.session.execute("\
        CREATE VIEW IF NOT EXISTS export_key_posts AS \
        SELECT p.post_id, \
            p.guid,\
            p.url as url, \
            max(pr.tfidf) AS tfidf1 \
        FROM posts p \
            INNER JOIN posts_representativeness pr on (pr.post_id = p.post_id)\
        GROUP BY p.post_id,\
                p.guid, \
                p.url;")

    def create_key_authors_view(self):
        self.session.execute("DROP VIEW IF EXISTS export_key_authors;")
        self.session.execute("CREATE VIEW IF NOT EXISTS export_key_authors AS \
        SELECT p.author AS author_name,\
            p.author_guid,\
            SUM(r.tfidf1) AS SumTFIDF,\
            MAX(r.tfidf1) AS MaxTFIDF\
        FROM posts p \
            JOIN \
            export_key_posts r ON (r.post_id = p.post_id)\
            where author_guid is not null and r.tfidf1 is not null\
         GROUP BY p.author_guid")

    def create_author_post_cite_view(self):
        self.session.execute("DROP VIEW IF EXISTS author_post_cite;")
        query = text("CREATE VIEW IF NOT EXISTS author_post_cite as \
                                SELECT DISTINCT posts.author_guid, post_citations.post_id_to \
                                FROM posts \
	                            INNER JOIN post_citations ON(post_citations.post_id_from = posts.post_id) \
	                            WHERE posts.author_guid is not null")
        self.session.execute(query)
        self.session.commit()

    def get_authors_topics(self, domain, min_posts_count):
        query = """
                SELECT author_topic_mapping.*
                FROM author_topic_mapping
                INNER JOIN author_guid_num_of_posts_view ON (author_guid_num_of_posts_view.author_guid = author_topic_mapping.author_guid)
                WHERE author_guid_num_of_posts_view.num_of_posts >= :min_posts_count
                AND domain = :domain
                """
        result = self.session.execute(query, params=dict(domain=domain, min_posts_count=min_posts_count))
        cursor = result.cursor
        author_topics_vectors = self.result_iter(cursor)

        author_guid_topics_vector = self._create_author_guid_topics_vector(author_topics_vectors)
        return author_guid_topics_vector

    def get_random_authors_topics(self, domain, min_posts_count):
        query = """
                SELECT author_topic_mapping.*
                FROM author_topic_mapping
                INNER JOIN random_authors_for_graphs ON (random_authors_for_graphs.author_guid = author_topic_mapping.author_guid)
                """
        result = self.session.execute(query, params=dict(domain=domain, min_posts_count=min_posts_count))
        cursor = result.cursor
        author_topics_vectors = self.result_iter(cursor)

        author_guid_topics_vector = self._create_author_guid_topics_vector(author_topics_vectors)
        return author_guid_topics_vector

    def _create_author_guid_topics_vector(self, author_topics_vectors):
        author_guid_topics_vector = {}
        for author_topics_vector in author_topics_vectors:
            author_guid = author_topics_vector[0]
            author_topic_vector = author_topics_vector[1:-1]

            author_guid_topics_vector[author_guid] = author_topic_vector
        return author_guid_topics_vector

    def create_author_guid_num_of_posts_view(self):
        self.session.execute("DROP VIEW IF EXISTS author_guid_num_of_posts_view;")
        query = """
                CREATE VIEW author_guid_num_of_posts_view as
                SELECT posts.author_guid, posts.domain, COUNT(*) as num_of_posts
                FROM posts
                GROUP BY 1,2
                HAVING num_of_posts >= 1
                ORDER BY 3 DESC
                """
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def get_posts_count_per_author(self, domain):
        query = text("SELECT authors.author_guid, COUNT(posts.post_id) "
                     "FROM authors "
                     "INNER JOIN posts ON (authors.author_guid = posts.author_guid) "
                     "WHERE authors.domain = :domain and authors.author_osn_id IS NOT NULL "
                     "GROUP BY authors.author_guid")

        result = self.session.execute(query, params=dict(domain=domain))
        cursor = result.cursor
        records = list(cursor.fetchall())
        return {record[0]: record[1] for record in records}

    #############################################################
    ###### Co-Citations View
    #############################################################
    def get_cocitations(self, min_number_of_cocited_posts):
        query = " SELECT a1.author_guid, a2.author_guid, count(a1.post_id_to) AS weight \
                  FROM author_post_cite a1 \
                  INNER JOIN author_post_cite a2  ON (a1.post_id_to = a2.post_id_to) \
                  WHERE a1.author_guid <> a2.author_guid \
                  GROUP BY a1.author_guid, a2.author_guid \
                  HAVING weight >=  :min_number_of_cocited_posts "

        print('starting get_cocitations query execution')
        result = self.session.execute(query, params=dict(min_number_of_cocited_posts=min_number_of_cocited_posts))
        print('passed get_cocitations query execution')

        cursor = result.cursor
        print('starting get_cocitations cursor fetchall')
        rows = self.result_iter(cursor)
        print('passed get_cocitations cursor fetchall')
        return rows

    ###########################################################
    ####### Citations View
    ###########################################################
    def get_citations(self, domain):
        query = " SELECT p_from.author_guid AS from_author_guid, p_to.author_guid AS to_author_guid,  count(*) AS num_citations \
                FROM post_citations AS p_cit \
                INNER JOIN posts AS p_from ON (p_cit.post_id_from = p_from.post_id) \
                INNER JOIN posts AS p_to ON (p_cit.post_id_to = p_to.post_id) \
                WHERE p_from.domain = :domain \
                AND p_to.domain = :domain \
                GROUP BY from_author_guid, to_author_guid "

        print('starting get_citations query execution')
        result = self.session.execute(query, params=dict(domain=domain))
        print('passed get_citations query execution')

        cursor = result.cursor
        print('starting get_citations cursor fetchall')
        rows = list(cursor.fetchall())
        print('passed get_citations cursor fetchall')
        return rows

    def get_random_author_guid_post_id_dictionary(self):
        query = """
                SELECT posts.author_guid, posts.post_id
                FROM posts
                INNER JOIN random_authors_for_graphs on random_authors_for_graphs.author_guid = posts.author_guid
                """
        return self._create_dictionary_by_query(query)

    def _create_dictionary_by_query(self, query):
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        records = self.result_iter(cursor)
        records = list(records)
        return {record[0]: record[1] for record in records}

    def get_post_id_random_author_guid_dictionary(self):

        query = """
                SELECT posts.post_id, posts.author_guid
                FROM posts
                INNER JOIN random_authors_for_graphs on random_authors_for_graphs.author_guid = posts.author_guid
                """
        return self._create_dictionary_by_query(query)

    ###########################################################
    # post representativeness
    ###########################################################
    def load_posts_representativeness_table(self):
        ufitf_data = self.get_ufitf_data()
        self.session.add_all(ufitf_data)
        self.session.commit()

    def create_posts_representativeness_entry(self, ufitf_value):
        return Posts_representativeness(
            post_id=format(list(ufitf_value.values())[0]),
            topic_id=int(list(ufitf_value.values())[1]),
            url=format(list(ufitf_value.values())[2]),
            how_many_times_cited_in_topic=int(list(ufitf_value.values())[3]),
            in_how_many_topics=int(list(ufitf_value.values())[4]),
            post_count=int(list(ufitf_value.values())[5]),
            tfidf=float(list(ufitf_value.values())[6]),
            tof=int(list(ufitf_value.values())[7]),
        )

    def get_ufitf_data(self):
        q = text(
            "select post_id, topic_id, url_to, how_many_times_cited_in_topic, in_how_many_topics, post_count, ufitf1, tof from tfidf")
        res = self.session.execute(q)
        return [self.create_posts_representativeness_entry(r) for r in res]

    def get_already_crawled_author_ids(self):
        query = "SELECT DISTINCT authors.author_osn_id " \
                "FROM authors " \
                "WHERE authors.author_osn_id IS NOT NULL " \
                "AND authors.missing_data_complementor_insertion_date IS NOT NULL"
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        ids = list(cursor.fetchall())
        already_crawled_author_ids = [r[0] for r in ids]
        return already_crawled_author_ids

    def get_bad_actor_retweeters_not_retrieved_from_vico(self):
        logging.info("get_bad_actor_retweeters_not_retrieved_from_vico")

        query = "SELECT authors.author_osn_id " \
                "FROM authors " \
                "INNER JOIN post_retweeter_connections on (authors.author_osn_id = post_retweeter_connections.retweeter_twitter_id) " \
                "WHERE authors.xml_importer_insertion_date IS NULL " \
                "AND authors.protected = 0 " \
                "AND authors.author_type = 'bad_actor'"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        ids = list(cursor.fetchall())
        twitter_authors_ids = [r[0] for r in ids]
        logging.info("Number of bad actor retweeters not retrieved from vico is: " + str(len(twitter_authors_ids)))
        return twitter_authors_ids

    def get_bad_actors_retweets_retrieved_by_vico(self):
        logging.info("get_bad_actors_retweets_retrieved_by_vico")

        results = self.session.query(Post).filter(or_(Post.content.like("%RT @annakiril3%"),
                                                      Post.content.like("%RT @LeviAvavilevi%"),
                                                      Post.content.like("%RT @benny_metanya%"),
                                                      Post.content.like("%RT @meggiewill5%"),
                                                      Post.content.like("%RT @amira_buzavgo%"),
                                                      Post.content.like("%RT @TAringthon%"))).all()

        return results

    def get_bad_actor_tweets_from_vico(self):
        logging.info("get_bad_actor_tweets_from_vico")
        '''
        SELECT *
        FROM posts
        WHERE (posts.content LIKE '%Youtube apps joins free Online TV channel in United kingdom%'
        OR posts.content LIKE '%Watch Internet TV, and Online TV for free!!%'
        OR posts.content LIKE '%Smart TV - all what we need to know!%'
        OR posts.content LIKE '%How to Stream Web Videos & Live TV to a Samsung Smart TV%'
        OR posts.content LIKE '%Free Internet TV - A Complete Guide For Canadians%'
        OR posts.content LIKE '%Smart TV vs. Media Streamer%' )
        AND posts.content NOT LIKE '%RT @%'
        '''

        results = self.session.query(Post).filter(
            or_(Post.content.like('%Youtube apps joins free Online TV channel in United kingdom%'),
                Post.content.like('%Watch Internet TV, and Online TV for free!!%'),
                Post.content.like('%Smart TV - all what we need to know!%'),
                Post.content.like('%How to Stream Web Videos & Live TV to a Samsung Smart TV%'),
                Post.content.like('%Free Internet TV - A Complete Guide For Canadians%'),
                Post.content.like('%Smart TV vs. Media Streamer%')),
            and_(not_(Post.content.like('%RT @%')))).all()

        return results

    def get_missing_authors_guid_not_marked_as_bad_actors(self, targeted_twitter_author_screen_names):
        #
        # This function retrieved all the authors who retweeted our posts and they are not marked as bad actors
        #

        logging.info("get_bad_actor_retweeters_not_retrieved_from_vico")

        query = "SELECT authors.author_guid " \
                "FROM authors " \
                "INNER JOIN posts ON (authors.author_guid = posts.author_guid) " \
                "WHERE (posts.content LIKE '%RT @annakiril3%' " \
                "OR posts.content LIKE '%RT @LeviAvavilevi%' " \
                "OR posts.content LIKE '%RT @benny_metanya%' " \
                "OR posts.content LIKE '%RT @meggiewill5%' " \
                "OR posts.content LIKE '%RT @amira_buzavgo%' " \
                "OR posts.content LIKE '%RT @TAringthon%') " \
                "AND authors.author_type IS NOT 'bad_actor'"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        twitter_author_guids = list(cursor.fetchall())
        author_guids = [r[0] for r in twitter_author_guids]
        logging.info("Number of missing bad actors that were not marked is: " + str(len(author_guids)))
        return author_guids

    def delete_acquired_authors(self):
        logging.info("delete_acquired_authors")
        query = 'DELETE ' \
                'FROM authors ' \
                'WHERE authors.author_type = "bad_actor" ' \
                'AND (authors.author_sub_type = "crowdturfer" ' \
                'OR authors.author_sub_type IS NULL ' \
                'OR authors.author_sub_type = "acquired")'
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def delete_manually_labeled_authors(self):
        logging.info("delete_manually_labeled_authors")
        query = 'DELETE ' \
                'FROM authors ' \
                'WHERE (authors.author_type = "bad_actor" OR authors.author_type = "good_actor")'
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def delete_posts_with_missing_authors(self):
        logging.info("detele_posts_with_missing_authors")
        query = ' DELETE ' \
                ' FROM posts' \
                ' WHERE (posts.author_guid NOT IN( ' \
                '    SELECT authors.author_guid' \
                '    FROM authors) ' \
                ' OR posts.author_guid IS NULL) '
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def get_posts_of_missing_authors(self):

        results = self.session.query(Post).filter(Post.author_guid == None).all()
        return results

    def get_missing_authors_tuples(self):
        query = ' SELECT DISTINCT(posts.author_guid), posts.author' \
                ' FROM posts' \
                ' WHERE (posts.author_guid NOT IN( ' \
                '    SELECT authors.author_guid' \
                '    FROM authors) ' \
                ' OR posts.author_guid IS NULL) '

        result = self.session.execute(query)
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        return tuples

    def get_author_screen_name_last_post_id(self):
        query = '''
        SELECT a.author_screen_name, p.post_osn_id, MIN(p.date) 
        FROM authors a, posts p
        WHERE a.author_guid = p.author_guid
        GROUP BY a.author_guid
        '''
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        author_screen_names_post_ids = list(cursor.fetchall())
        # return cursor
        return self._get_author_screen_name_tweet_id_dict(author_screen_names_post_ids)

    def get_author_screen_name_first_post_id(self):
        query = '''
        SELECT a.author_screen_name, p.post_osn_id, MAX(p.date) 
        FROM authors a, posts p
        WHERE a.author_guid = p.author_guid
        GROUP BY a.author_guid
        '''
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        author_screen_names_post_ids = list(cursor.fetchall())
        # return cursor
        return self._get_author_screen_name_tweet_id_dict(author_screen_names_post_ids)

    def _get_author_screen_name_tweet_id_dict(self, author_screen_names_post_ids):
        author_screen_name_tweet_id = {}
        for author_screen_name, post_osn_id, date in author_screen_names_post_ids:
            author_screen_name_tweet_id[author_screen_name] = post_osn_id
        return author_screen_name_tweet_id

    def get_author_screen_names_and_number_of_posts(self, num_of_minimal_posts):

        logging.info("get_author_screen_names_for_timelines")

        query = "SELECT authors.author_screen_name, COUNT(authors.author_guid) " \
                "FROM authors " \
                "INNER JOIN posts ON(authors.author_guid = posts.author_guid) " \
                "WHERE authors.domain = 'Microblog'  " \
                "AND authors.author_osn_id IS NOT NULL " \
                "AND authors.protected = 0 " \
                "AND authors.timeline_overlap_insertion_date IS NULL " \
                "GROUP BY authors.author_guid " \
                "HAVING COUNT(authors.author_guid) < :num_of_minimal_posts"

        query = text(query)
        result = self.session.execute(query, params=dict(num_of_minimal_posts=num_of_minimal_posts))
        cursor = result.cursor
        # return cursor
        osn_author_screen_names_and_number_of_posts = list(cursor.fetchall())
        return osn_author_screen_names_and_number_of_posts

    def assign_manually_labeled_authors(self):
        self.assign_private_profiles()
        self.assign_company_profiles()
        self.assign_bot_profiles()
        self.assign_news_feed_profiles()
        self.assign_spammer_profiles()

    def assign_private_profiles(self):
        logging.info("assign_private_profiles")
        sql_script = open('DB/scripts/assign_private_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def assign_company_profiles(self):
        logging.info("assign_company_profiles")
        sql_script = open('DB/scripts/assign_company_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def assign_news_feed_profiles(self):
        logging.info("assign_news_feed_profiles")
        sql_script = open('DB/scripts/assign_news_feed_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def assign_spammer_profiles(self):
        logging.info("assign_spammer_profiles")
        sql_script = open('DB/scripts/assign_spammer_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def assign_bot_profiles(self):
        logging.info("assign_bot_profiles")
        sql_script = open('DB/scripts/assign_bot_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def assign_crowdturfer_profiles(self):
        logging.info("assign_crowdturfer_profiles")
        sql_script = open('DB/scripts/assign_crowdturfer_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def assign_acquired_profiles(self):
        logging.info("assign_acquired_profiles")
        sql_script = open('DB/scripts/assign_acquired_profiles.txt', 'r')
        query = sql_script.read()
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def get_all_acquired_crowdturfer_authors(self):
        logging.info("get_all_acquired_crowdturfer_authors")
        query = 'SELECT * ' \
                'FROM authors ' \
                'WHERE (authors.author_type = "bad_actor" ' \
                'AND (authors.author_sub_type = "acquired" OR authors.author_sub_type = "crowdturfer" OR authors.author_sub_type IS NULL));'
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        data = list(cursor.fetchall())
        authors = [row[0] for row in data]
        return authors

    def get_all_manually_labeled_bad_actors(self):
        logging.info("get_all_manually_labeled_bad_actors")
        query = 'SELECT * ' \
                'FROM authors ' \
                'WHERE (authors.author_type = "bad_actor" ' \
                'AND authors.author_sub_type IS NOT NULL);'
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        data = list(cursor.fetchall())
        authors = [row[0] for row in data]
        return authors

    def get_all_unlabeled_authors(self):
        logging.info("get_all_unlabeled_authors")
        query = "SELECT * " \
                "FROM authors " \
                "WHERE authors.author_type IS NULL " \
                "AND authors.domain = 'Microblog'"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        data = list(cursor.fetchall())
        authors = [row[0] for row in data]
        return authors

    def update_bad_actor_from_timeline_overlaping(self, potential_bad_actors):
        logging.info("update_bad_actor_from_timeline_overlaping")
        query = 'UPDATE authors ' \
                'SET author_type = "bad_actor", author_sub_type = "acquired", timeline_overlap_insertion_date = :insertion_date ' \
                'WHERE authors.name IN ' + "('" + "','".join(map(str, potential_bad_actors)) + "')"
        query = text(query)
        date = str(get_current_time_as_string())
        self.session.execute(query, params=dict(insertion_date=date))
        self.session.commit()

    def update_authors_type_by_author_names(self, authors_name, author_type):
        logging.info("update_authors_type_by_author_names")
        query = 'UPDATE authors ' \
                'SET author_type = :author_type ' \
                'WHERE authors.name IN ' + "('" + "','".join(map(str, authors_name)) + "')"
        query = text(query)
        self.session.execute(query, params=dict(author_type=author_type))
        self.session.commit()

    def create_authors_index(self):
        logging.info("create_authors_index")
        query = "CREATE INDEX IF NOT EXISTS idx_authors " \
                "ON authors (domain, author_osn_id)"

        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def create_posts_index(self):
        logging.info("create_authors_index")
        query = "CREATE INDEX IF NOT EXISTS idx_posts " \
                "ON posts (author_guid)"

        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def get_words_with_highest_probability(self):
        query = "select * " \
                "from topic_terms_view " \
                "order by topic_id asc,probability desc ;"
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        data = list(cursor.fetchall())
        result = {}
        current_topic = str(data[0][0])
        get_from_topic = 0
        for row in data:

            if str(row[0]) != current_topic:
                get_from_topic = 0
                current_topic = str(row[0])

            if current_topic not in result:
                result[current_topic] = []

            if str(row[0]) == current_topic and get_from_topic <= 10:
                get_from_topic += 1
                result[current_topic].append((row[1], row[2]))

        return result

    def get_author_timelines_by_min_num_of_posts(self, domain, min_num_of_posts):
        query = """
                select
                a.author_guid,
                p.content,
                a.author_type
                from authors as a
                inner join posts as p on (a.author_guid = p.author_guid)
                where a.domain= :domain
                and a.author_guid in (  select
                                        posts.author_guid
                                        from posts
                                        where domain = :domain
                              group by posts.author_guid
                              having count(posts.author_guid) >= :min_num_of_posts)
                """
        query = text(query)

        result = self.session.execute(query, params=dict(domain=domain, min_num_of_posts=min_num_of_posts))
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        return tuples

    def get_author_connections_by_connection_type(self, connection_type):
        query = """
                SELECT author_connections.source_author_guid, author_connections.destination_author_guid, author_connections.weight
                FROM author_connections
                WHERE author_connections.connection_type = :connection_type
                """
        query = text(query)

        result = self.session.execute(query, params=dict(connection_type=connection_type))
        cursor = result.cursor
        generator = self.result_iter(cursor)
        return generator

    def get_labeled_authors_by_domain(self, domain, targeted_class_field_name):
        query = """
                SELECT authors.author_guid, authors.{}
                FROM authors
                WHERE authors.domain = domain
                AND authors.author_type IS NOT NULL
                """.format(targeted_class_field_name)
        query = text(query)

        result = self.session.execute(query, params=dict(domain=domain))
        cursor = result.cursor
        generator = self.result_iter(cursor)
        return generator

    def get_labeled_author_connections_by_connection_type(self, connection_type):
        query = """
                SELECT author_connections.source_author_guid, author_connections.destination_author_guid, author_connections.weight
                FROM author_connections
                INNER JOIN authors as a1 ON (a1.author_guid = author_connections.source_author_guid)
                INNER JOIN authors as a2 ON (a2.author_guid = author_connections.destination_author_guid )
                WHERE author_connections.connection_type = :connection_type
                AND a1.author_type IS NOT NULL
                AND a2.author_type IS NOT NULL
                """
        query = text(query)

        result = self.session.execute(query, params=dict(connection_type=connection_type))
        cursor = result.cursor
        generator = self.result_iter(cursor)
        return generator

        return cursor

    def get_labeled_bad_actors_timelines_temp(self):
        query = """ select
                               a.name,
                               p.content,
                               a.author_sub_type
                            from
                                authors as a
                            inner join
                                posts as p on (a.author_guid = p.author_guid)
                            where
                                 a.domain= 'Microblog'
                            and (a.author_type = 'bad_actor' or a.author_type = 'good_actor')
                            and a.author_sub_type in ('bot','spammer','crowdturfer','acquired','news_feed','private','company' )
                            and a.author_guid in (  select
                                                        posts.author_guid
                                                    from posts
                                                    where domain = 'Microblog'
                                              group by posts.author_guid
                                              having count(posts.author_guid) >= 100)
                    """
        query = text(query)

        result = self.session.execute(query)
        cursor = result.cursor

        return cursor

    def get_authors_and_tweets_ids_from_temporal_table(self):
        '''
        :return: a list of Twitter statuses id
        '''

        q = " select post_id, author_id from ( "
        for i in range(3, 203):
            q += "select field" + str(i) + " as post_id, twitter_id as author_id from honeypot where field" + str(
                i) + " is not null \n "
            if i < 202:
                q += " union "
        q += ") \n "
        q += " where post_id not in (select post_osn_id from posts) " \
             " and  post_id not in (select tweet_id from deleted_tweets) " \
             " and author_id not in (select author_osn_id from authors where protected = 1 or is_suspended_or_not_exists = 1)"
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())
        result = []
        for rec in records:
            result.append((rec[0], rec[1]))

        return result

    def save_author_features(self, authors_features):
        print('\n Beginning merging author_features objects')
        counter = 0
        if authors_features:
            for author_features_row in authors_features:
                counter += 1
                self.update_author_features(author_features_row)
                if counter == 100:
                    print("\r " + "merging author-features objects", end="")
                    self.commit()
                    counter = 0
            if counter != 0:
                self.commit()
        print('Finished merging author_features objects')

    def create_topic_terms_view(self):
        print("create_topic_terms_view ")
        query = """
                create view IF NOT EXISTS topic_terms_view as
		        select topic_id, t2.description, probability
                from topics t1 inner join terms t2 on t1.term_id = t2.term_id
                """
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    def get_cooperated_authors(self, targeted_twitter_author_names, domain):
        query = """
                SELECT DISTINCT authors.author_guid
                FROM authors
                INNER JOIN posts ON (authors.author_guid = posts.author_guid)
                WHERE authors.domain = :domain
                AND authors.author_type IS NOT 'bad_actor'
                AND ( """

        targeted_twitter_authors_count = len(targeted_twitter_author_names)
        query += "posts.content LIKE '%RT @" + targeted_twitter_author_names[0] + "%' "

        for i in range(1, targeted_twitter_authors_count):
            query += "OR posts.content LIKE '%RT @" + targeted_twitter_author_names[i] + "%' "

        query += ")"

        query = text(query)

        result = self.session.execute(query, params=dict(domain=domain))
        cursor = result.cursor

        return cursor

    def create_post_topic_mapping_post_id_index(self):
        logging.info("create_post_topic_mapping_post_id_index")
        query = "CREATE INDEX IF NOT EXISTS create_post_topic_mapping_post_id_index " \
                "ON post_topic_mapping (post_id)"

    def create_posts_post_id_index(self):
        logging.info("create_posts_post_id_index")
        query = "CREATE INDEX IF NOT EXISTS create_posts_post_id_index " \
                "ON posts (posts)"

    def get_unlabeled_predictions(self):
        query = """
                SELECT unlabeled_predictions.AccountPropertiesFeatureGenerator_author_screen_name,
                        unlabeled_predictions.predicted,
                        unlabeled_predictions.prediction
                FROM unlabeled_predictions
                """
        query = text(query)

        result = self.session.execute(query)
        cursor = result.cursor

        return cursor

    def drop_unlabeled_predictions(self, predictions_table_name):
        query = "DROP TABLE IF EXISTS " + predictions_table_name + ";"
        query = text(query)
        self.session.execute(query)
        self.session.commit()

    ###########################################################
    ####### Get distance features
    ###########################################################

    def get_distance_features(self):
        query = " SELECT author_guid \
                       FROM author_features  \
                       WHERE attribute_name like '%min_dist_to%'  \
                   or attribute_name like '%mean_dist_to%' "

        result = self.session.execute(query)
        cursor = result.cursor
        rows = list(cursor.fetchall())
        return rows

    def insert_or_update_authors_from_posts(self, domain, author_classify_dict, author_prop_dict):
        authors_to_update = []
        posts = self.session.query(Post).filter(Post.domain == domain).all()
        author_dict = self.get_author_dictionary()
        logging.info("Insert or update_authors from app importer")
        logging.info("total Posts: " + str(len(posts)))
        i = 1
        for post in posts:
            msg = "\r Insert or update posts: [{}".format(i) + "/" + str(len(posts)) + ']'
            print(msg, end="")
            i += 1
            author_guid = post.author_guid
            if author_guid in author_dict:
                continue

            # domain = post.domain

            # if not self.is_author_exists(author_guid, domain):
            author = Author()
            author_name = post.author
            author.name = author_name
            author.author_screen_name = post.author
            author.domain = post.domain
            author.author_guid = author_guid

            if author_name in author_classify_dict:
                author.author_type = author_classify_dict[author_name]

            post_type = post.post_type
            if post_type is not None:
                targeted_classes = post_type.split('/')
                author_sub_type = targeted_classes[0]
                if author_sub_type is not None:
                    author.author_sub_type = author_sub_type

            if author_guid in author_prop_dict:
                for key, value in author_prop_dict[author_guid].items():
                    setattr(author, key, value)

            authors_to_update.append(author)
            author_dict[author_guid] = author

        if len(posts) != 0: print("")
        # self.add_authors(authors_to_update)
        self.session.bulk_save_objects(authors_to_update)
        self.session.commit()

    def get_posts_filtered_by_domain(self, domain):
        entries = self.session.query(Post).filter(Post.domain == domain).all()
        return entries

    def get_instagram_posts_without_comments(self):
        connection_type = 'post_comment_connection'
        query = text("""
                        SELECT *
                        FROM posts
                        WHERE posts.post_type = 'post'
                        AND posts.domain = 'Instagram'
                        AND posts.retweet_count > 0
                        AND posts.post_id  NOT IN (SELECT source_author_guid
                                                   FROM author_connections
                                                   WHERE connection_type = :connection_type)
                     """)

        result = self.session.execute(query, params=dict(connection_type=connection_type))
        posts_dicts = list(map(dict, result))
        posts = [Post(**post_dict) for post_dict in posts_dicts]
        return posts

    def get_author_guids(self):
        result = self.session.query(Author.author_guid).all()
        ids = [res[0] for res in result]
        return ids

    def delete_anchor_authors(self):
        query = """
                DELETE
                FROM anchor_authors
                """
        query = text(query)
        self.session.execute(query)
        self.commit()

    def insert_anchor_author(self, author_guid, author_type):
        anchor_author = AnchorAuthor(author_guid, author_type)
        self.session.merge(anchor_author)
        self.session.commit()

    def get_anchor_authors(self):
        query = """  SELECT author_guid, author_type
                     FROM anchor_authors  """
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        return cursor

    def get_random_authors_for_graphs(self):
        query = """
                SELECT author_guid, author_type
                FROM random_authors_for_graphs
                """
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        random_authors_for_graphs = self.result_iter(cursor)
        return random_authors_for_graphs

    # def get_author_types(self, domain):
    #     author_type = {}
    #     query = "SELECT author_guid, author_type FROM authors where author_guid is not null and domain={0}".format(domain)
    #     query = text(query)
    #     result = self.session.execute(query)
    #     authors = self.result_iter(result.cursor)
    #     for author in authors:
    #         author_type[author[0]]=author[1]
    #     return author_type

    def create_author_dictionaries(self, index_field_for_predictions, domain):
        labeled_author_dict = {}
        unlabeled_author_dict = {}
        # author_guid - author_screen_name
        unlabeled_author_guid_index_field_dict = {}
        query = "SELECT author_guid, {0},  author_type FROM authors where author_guid is not null and domain='{1}'".format(
            index_field_for_predictions, domain)
        # query = """
        #         SELECT author_guid, author_type FROM authors where author_guid is not null and domain = '" +domain + "'
        #         """
        query = text(query)
        result = self.session.execute(query)
        authors = self.result_iter(result.cursor)
        for author in authors:
            author_guid = author[0]
            index_field_for_predictions = author[1]
            targeted_class = author[2]

            if targeted_class is not None:
                labeled_author_dict[author_guid] = targeted_class
                print("{0} - {1}".format(author_guid, targeted_class))
            else:
                unlabeled_author_dict[author_guid] = targeted_class
                unlabeled_author_guid_index_field_dict[author_guid] = index_field_for_predictions
        return labeled_author_dict, unlabeled_author_dict, unlabeled_author_guid_index_field_dict

    def get_author_guid_by_targeted_field_name_and_targeted_class(self, targeted_field_name, targeted_class):
        query = "SELECT authors.author_guid FROM authors WHERE authors"
        query += "." + targeted_field_name + " = " + "'" + targeted_class + "'"

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        return cursor

    def create_author_feature(self, author_guid, attribute_name, attribute_value):
        author_feature = AuthorFeatures()

        author_feature.author_guid = author_guid
        author_feature.attribute_name = attribute_name
        author_feature.attribute_value = str(attribute_value)
        author_feature.window_start = self._window_start
        author_feature.window_end = self._window_end

        msg = '\r adding ' + 'author_guid:' + author_guid + ' attribute_name: ' + attribute_name + ' attribute_value: ' + str(
            attribute_value)
        print(msg, end="")

        return author_feature

    def delete_anchor_author_features(self):
        query = """
                DELETE
                FROM author_features
                WHERE author_features.author_guid IN (
	              SELECT anchor_authors.author_guid
	              FROM anchor_authors
                )
                """
        query = text(query)
        self.session.execute(query)
        self.commit()

    def create_temp_author_connections(self, source_author_id, destination_author_ids, author_connection_type,
                                       insertion_date):
        print("---create_temp_author_connections---")
        author_connections = []
        for destination_author_id in destination_author_ids:
            author_connection = self.create_temp_author_connection(source_author_id, destination_author_id,
                                                                   author_connection_type, insertion_date)
            author_connections.append(author_connection)

        return author_connections

    def create_temp_author_connection(self, source_author_id, destination_author_id, connection_type, insertion_date):
        temp_author_connection = TempAuthorConnection()
        msg = '\r "Temp author connection: source -> ' + str(source_author_id) + ', dest -> ' + str(
            destination_author_id) + ', connection type = ' + connection_type
        print(msg, end="")
        # print("Temp author connection: source -> " + str(source_author_id) + ", dest -> " + str(
        #     destination_author_id) + ", connection type = " + connection_type)
        temp_author_connection.source_author_osn_id = source_author_id
        temp_author_connection.destination_author_osn_id = destination_author_id
        temp_author_connection.connection_type = str(connection_type)
        temp_author_connection.insertion_date = insertion_date

        return temp_author_connection

    def get_temp_author_connections(self):
        query = """
                SELECT temp_author_connections.source_author_osn_id, temp_author_connections.destination_author_osn_id,
                       temp_author_connections.connection_type, temp_author_connections.insertion_date
                FROM temp_author_connections
                """
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        return cursor

    def delete_temp_author_connections(self, temp_author_connections):
        total = len(temp_author_connections)
        current = 0
        for author_connection in temp_author_connections:
            current += 1
            msg = '\r adding ' + str(current) + ' of ' + str(total) + ' author_connections'
            print(msg, end="")
            self.delete_temp_author_connection(author_connection)
        self.session.commit()

    def delete_temp_author_connection(self, temp_author_connection):
        query = """
                DELETE FROM temp_author_connections
                WHERE source_author_osn_id = :source_id
                AND destination_author_osn_id = :destination_id
                AND connection_type = :connection_type
        """
        query = text(query)
        self.session.execute(query, params=dict(source_id=temp_author_connection.source_author_osn_id,
                                                destination_id=temp_author_connection.destination_author_osn_id,
                                                connection_type=temp_author_connection.connection_type))

    def create_post_retweeter_connections(self, post_id, retweeter_ids):
        post_retweeter_connections = []
        retweeter_connection_type = str("post_retweeter")
        for retweeter_id in retweeter_ids:
            post_retweeter_connection = self.create_post_retweeter_connection(post_id, retweeter_id,
                                                                              retweeter_connection_type)
            post_retweeter_connections.append(post_retweeter_connection)

        return post_retweeter_connections

    def create_post_retweeter_connection(self, post_id, retweeter_id, connection_type):
        post_retweeter_connection = PostRetweeterConnection()

        post_retweeter_connection.post_osn_id = post_id
        post_retweeter_connection.retweeter_twitter_id = retweeter_id
        post_retweeter_connection.connection_type = str(connection_type)
        post_retweeter_connection.insertion_date = str(get_current_time_as_string())

        return post_retweeter_connection

    def convert_temp_author_connections_to_author_connections(self, domain):
        cursor = self.get_temp_author_connections()
        temp_author_connection_tuples = self.result_iter(cursor)

        author_osn_id_author_guid_dict = self.create_author_osn_id_author_guid_dictionary(domain)

        author_connections = []
        already_converted_temp_author_connections = []
        for temp_author_connection in temp_author_connection_tuples:
            source_author_osn_id = temp_author_connection[0]
            destination_author_osn_id = temp_author_connection[1]
            connection_type = temp_author_connection[2]
            insertion_date = temp_author_connection[3]

            if source_author_osn_id in author_osn_id_author_guid_dict and destination_author_osn_id in author_osn_id_author_guid_dict:
                source_author_guid = author_osn_id_author_guid_dict[source_author_osn_id]
                destination_author_guid = author_osn_id_author_guid_dict[destination_author_osn_id]

                author_connection = self.create_author_connection(source_author_guid, destination_author_guid, 0,
                                                                  connection_type, insertion_date)
                already_convert_temp_author_connection = self.create_temp_author_connection(source_author_osn_id,
                                                                                            destination_author_osn_id,
                                                                                            connection_type,
                                                                                            insertion_date)

                author_connections.append(author_connection)
                already_converted_temp_author_connections.append(already_convert_temp_author_connection)
        self.save_author_connections(author_connections)
        self.delete_temp_author_connections(already_converted_temp_author_connections)

    def create_author_osn_id_author_guid_dictionary(self, domain):
        author_osn_id_author_guid_dict = {}
        authors = self.get_authors_by_domain(domain)
        for author in authors:
            author_osn_id = author.author_osn_id
            author_guid = author.author_guid
            author_osn_id_author_guid_dict[author_osn_id] = author_guid
        return author_osn_id_author_guid_dict

    def get_topic_with_maximal_posts(self):
        query = """
                SELECT res.topic_id, MAX(res.post_count)
                FROM (
                SELECT topic_stats.topic_id, topic_stats.post_count
                FROM topic_stats
                GROUP BY 1
                ORDER BY 2 DESC
                ) res
                """

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        generator = self.result_iter(cursor)
        for tuple in generator:
            return tuple

    def get_top_terms_by_topic_id(self, topic_id):
        query = """
                SELECT topics.term_id, terms.description, topics.probability
                FROM topics
                INNER JOIN terms ON (terms.term_id = topics.term_id)
                WHERE topics.topic_id = {}
                ORDER BY 3 DESC
                LIMIT 100
                """.format(topic_id)

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        generator = self.result_iter(cursor)
        top_terms = [term[1] for term in generator]
        return top_terms

    def get_top_10_terms_by_topic_id(self, topic_id):
        query = """
                SELECT topics.term_id, terms.description, topics.probability
                FROM topics
                INNER JOIN terms ON (terms.term_id = topics.term_id)
                WHERE topics.topic_id = {}
                ORDER BY 3 DESC
                LIMIT 10
                """.format(topic_id)

        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        generator = self.result_iter(cursor)
        top_terms = [term[1] for term in generator]
        return top_terms

    def create_post_from_tweet_data(self, tweet_data, domain):
        author_name = tweet_data.user.screen_name
        tweet_author_guid = compute_author_guid_by_author_name(author_name)
        tweet_post_twitter_id = str(tweet_data.id)
        tweet_url = generate_tweet_url(tweet_post_twitter_id, author_name)
        tweet_creation_time = tweet_data.created_at
        tweet_str_publication_date = extract_tweet_publiction_date(tweet_creation_time)
        tweet_guid = compute_post_guid(post_url=tweet_url, author_name=author_name,
                                       str_publication_date=tweet_str_publication_date)

        media_path = None
        if tweet_data.media is not None:
            if tweet_data.media[0] is not None:
                media_url = tweet_data.media[0].media_url
                media_path = str(media_url)
        post = Post(guid=tweet_guid, post_id=tweet_guid, url=str(tweet_url),
                    date=str_to_date(tweet_str_publication_date),
                    title=str(tweet_data.text), content=str(tweet_data.text),
                    post_osn_id=tweet_post_twitter_id,
                    author=str(author_name), author_guid=str(tweet_author_guid),
                    domain=str(domain),
                    media_path=media_path,
                    retweet_count=str(tweet_data.retweet_count),
                    favorite_count=str(tweet_data.favorite_count),
                    timeline_importer_insertion_date=str(get_current_time_as_string()))
        return post

    def get_max_topic(self):
        query = """
                SELECT MAX(topics.topic_id)
                FROM topics
                """
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        rows = cursor.fetchall()
        max_topic = rows[0][0]
        return max_topic

    def _get_top_terms_by_topic_id(self, topic_id, num_of_top_terms):
        query = """
                SELECT topics.topic_id, terms.description, topics.probability
                FROM topics
                INNER JOIN terms on (topics.term_id = terms.term_id)
                WHERE topics.topic_id = {0}
                ORDER BY topics.probability DESC
                LIMIT {1}
                """.format(topic_id, num_of_top_terms)
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = cursor.fetchall()

        top_terms = [tuple[1] for tuple in tuples]
        return top_terms

    def _randomize_authors(self, min_posts_count, domain, num_of_random_authors):
        query = """
                SELECT authors.author_guid, authors.author_type
                FROM authors
                INNER JOIN author_guid_num_of_posts_view ON (author_guid_num_of_posts_view.author_guid = authors.author_guid)
                WHERE author_guid_num_of_posts_view.num_of_posts >= :min_posts_count
                AND authors.domain = :domain
                ORDER BY RANDOM()
                LIMIT :num_of_random_authors
                """
        query = text(query)
        result = self.session.execute(query, params=dict(min_posts_count=min_posts_count, domain=domain,
                                                         num_of_random_authors=num_of_random_authors))
        cursor = result.cursor
        randomized_authors_for_graph = self.result_iter(cursor)
        return randomized_authors_for_graph

    def _create_randomized_author_for_graph(self, author_guid, author_type):
        randomized_author_for_graph = RandomAuthorForGraph()
        randomized_author_for_graph.author_guid = author_guid
        randomized_author_for_graph.author_type = author_type
        return randomized_author_for_graph

    def randomize_authors_for_graph(self, min_posts_count, domain, num_of_random_authors_for_graph):
        randomized_authors = []
        randomized_authors_for_graph = self._randomize_authors(min_posts_count, domain, num_of_random_authors_for_graph)
        for author_guid, author_type in randomized_authors_for_graph:
            randomized_author_for_graph = self._create_randomized_author_for_graph(author_guid, author_type)
            randomized_authors.append(randomized_author_for_graph)

    def deleteTopics(self, window_start=None):
        if window_start:
            self.session.query(Topic).filter(Topic.window_start == window_start).delete()
            self.session.query(Post_to_topic).filter(Post_to_topic.window_start == window_start).delete()
        else:
            self.session.query(Topic).delete()
            self.session.query(Post_to_topic).delete()
        self.session.commit()

    def addTopics(self, topics):
        for topic in topics:
            self.addTopic(topic)
        self.session.commit()

    def addTopic(self, topic):
        self.session.merge(topic)

    def addPostTopicMapping(self, topic_mapping):
        self.session.merge(topic_mapping)

    def addPostTopicMappings(self, post_topic_mappings):
        for i, topic_mapping in enumerate(post_topic_mappings):
            self.addPostTopicMapping(topic_mapping)
            if i % 100 == 0:
                msg = "\rAdd post topic mappings {0}/{1}".format(str(i), str(len(post_topic_mappings)))
                print(msg, end='')
        msg = "\rAdd post topic mappings {0}/{1}".format(str(len(post_topic_mappings)), str(len(post_topic_mappings)))
        print(msg, end='')
        self.session.commit()

    def add_terms(self, terms):
        for term in terms:
            self.add_term(term)
        self.session.commit()

    def add_term(self, term):
        self.session.merge(term)

    def add_topic_items(self, topic_items):
        for topic_item in topic_items:
            self.add_topic_item(topic_item)
        self.session.commit()

    def add_topic_item(self, topic_item):
        self.session.merge(topic_item)

    def create_author_topic_mapping_table(self, number_of_topics):
        query = """
            CREATE TABLE IF NOT EXISTS author_topic_mapping (
            author_guid text NOT NULL,
            {0}
            CONSTRAINT PK_Person PRIMARY KEY (author_guid)
            FOREIGN KEY (author_guid) REFERENCES authors(author_guid));
        """
        topics = ""
        for i in range(number_of_topics):
            topics += "'{0}' int NOT NULL,\n".format(i)
        query = query.format(topics)
        query = text(query)
        self.session.execute(query)

    def insert_into_author_toppic_mapping(self, author_guid, author_mapping):

        query = """
                    INSERT INTO author_topic_mapping 
                    VALUES ('{0}',{1});
                """
        author_mapping = ','.join([str(m) for m in author_mapping])

        query = query.format(author_guid, author_mapping)
        query = text(query)
        self.session.execute(query)

    def insert_into_author_toppic_mappings(self, mappings):
        if len(mappings) > 0:
            author_mappings = []
            author_guids = []
            for author_guid, author_mapping in mappings:
                mapping_tamplate = "('{0}',{1})"
                author_mapping = ','.join([str(m) for m in author_mapping])
                author_mappings.append(mapping_tamplate.format(author_guid, author_mapping))
                author_guids.append("'{0}'".format(author_guid))

            self.delete_author_topic_mapping_by_author_guids(author_guids)

            for i in range(int(len(mappings) / 10000 + 1)):
                author_topic_mapping_count = str(min((i + 1) * 10000, len(mappings)))
                print('\r add author_topic_mappings {}/{}'.format(author_topic_mapping_count, len(mappings)), end='')
                query = """         
                            INSERT INTO author_topic_mapping 
                            VALUES {0};
                            """
                query = query.format(',\n'.join(author_mappings[i * 10000: (i + 1) * 10000]))
                query = text(query)
                self.session.execute(query)
                self.session.commit()
            print()

    def delete_author_topic_mapping_by_author_guids(self, author_guids):
        query = """
                    DELETE FROM author_topic_mapping
                    WHERE author_guid IN ({0});
                """
        query = query.format(','.join(author_guids))
        query = text(query)
        self.session.execute(query)

    def delete_terms(self):
        self.session.query(Term).delete()
        self.session.commit()

    def delete_post_topic_mapping(self):
        self.session.query(PostTopicMapping).delete()
        self.session.commit()

    def delete_author_topic_mapping(self):
        query = """
                DROP TABLE IF EXISTS author_topic_mapping;
                """
        query = text(query)
        self.session.execute(query)

    def get_terms(self):
        return self.session.query(Term).all()

    def get_author_topic_mapping(self):
        query = """
                SELECT * FROM author_topic_mapping
                """
        query = text(query)
        result = self.session.execute(query)
        return result.cursor.fetchall()

    def get_post_topic_mapping(self):
        return self.session.query(PostTopicMapping).all()

    def get_number_of_topics(self):
        return self.session.execute("select count(distinct( topic_id)) from topics").scalar()

    @staticmethod
    def create_post_topic_mapping_obj(max_topic_probability, post_id):
        ptm = PostTopicMapping()
        ptm.post_id = post_id
        ptm.max_topic_dist = float(max_topic_probability[1])
        ptm.max_topic_id = int(max_topic_probability[0])
        return ptm

    @staticmethod
    def create_topic_item(topic_id, term_id, probability):
        topic_obj = Topic()
        topic_obj.topic_id = topic_id
        topic_obj.term_id = term_id
        topic_obj.probability = probability
        return topic_obj

    @staticmethod
    def create_term(term_id, term_description):
        term = Term()
        term.term_id = term_id
        term.description = term_description
        return term

    def get_targeted_articles(self):
        targetd_articles = self.session.query(Target_Article).all()
        return targetd_articles

    def get_targeted_article_items(self):
        targetd_articles = self.session.query(Target_Article_Item).all()
        return targetd_articles

    def get_text_images(self):
        text_images = self.session.query(Text_From_Image).all()
        return text_images

    def get_authors_with_media(self):
        query = """SELECT authors.name, authors.media_path FROM authors
                    WHERE  authors.media_path IS NOT NULL"""
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        return tuples

    def get_authors_and_image_tags(self):
        query = """SELECT * FROM image_tags"""
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        return tuples

    def get_post_id_to_author_guid_mapping(self):
        query = """
                        SELECT posts.author_guid, posts.post_id 
                        FROM posts
                        """
        result = self.session.execute(query)
        cursor = result.cursor
        records = self.result_iter(cursor)
        records = list(records)

        return {record[1]: record[0] for record in records}

    def get_author_guid_word_embedding_vector_dict(self, word_embedding_table_name, table_name, targeted_field_name,
                                                   word_embedding_type):
        query = self._get_author_guid_word_embedding_vector_full_query(word_embedding_table_name, table_name,
                                                                       targeted_field_name, word_embedding_type)
        result = self.session.execute(query, params=dict(word_embedding_table_name=word_embedding_table_name,
                                                         table_name=table_name, targeted_field_name=targeted_field_name,
                                                         word_embedding_type=word_embedding_type))
        return self._create_author_guid_word_embedding_vector_dict_by_query(result)

    def get_random_author_guid_word_embedding_vector_dict(self, table_name, targeted_field_name, word_embedding_type,
                                                          num_of_random_authors_for_graph):
        query = self._get_random_author_guid_word_embedding_vector_full_query(table_name, targeted_field_name,
                                                                              word_embedding_type,
                                                                              num_of_random_authors_for_graph)
        result = self.session.execute(query, params=dict(word_embedding_table_name=word_embedding_table_name,
                                                         table_name=table_name, targeted_field_name=targeted_field_name,
                                                         word_embedding_type=word_embedding_type,
                                                         num_of_random_authors_for_graph=num_of_random_authors_for_graph))
        return self._create_author_guid_word_embedding_vector_dict_by_query(result)

    def _get_word_embeddings_types(self, author_word_embeddings="author_word_embeddings"):
        query = "SELECT word_embedding_type from %s GROUP BY 1" % author_word_embeddings
        result = self.session.execute(query).fetchall()
        parsed = [col[0] for col in result]
        return parsed

    def _get_author_guid_word_embedding_vector_full_query(self, word_embedding_table_name, table_name,
                                                          targeted_field_name, word_embedding_type):
        query = """
                SELECT *
                FROM {0}
                WHERE table_name = '{1}'
                AND targeted_field_name = '{2}'
                AND word_embedding_type = '{3}'
                AND author_id IS NOT NULL
                """.format(word_embedding_table_name, table_name, targeted_field_name, word_embedding_type)
        return query

    def _get_random_author_guid_word_embedding_vector_full_query(self, word_embedding_table_name, table_name,
                                                                 targeted_field_name, word_embedding_type,
                                                                 num_of_random_authors_for_graph):
        query = """
                SELECT *
                FROM {0}
                WHERE table_name = '{1}'
                AND targeted_field_name = '{2}'
                AND word_embedding_type = '{3}'
                AND author_id IS NOT NULL
                LIMIT {4}
                """.format(word_embedding_table_name, table_name, targeted_field_name, word_embedding_type,
                           num_of_random_authors_for_graph)
        return query

    def _create_author_guid_word_embedding_vector_dict_by_query(self, result):
        cursor = result.cursor
        records = self.result_iter(cursor)
        # records = list(records)
        return self.create_author_guid_word_embedding_dict_by_recoreds(records)

    def create_author_guid_word_embedding_dict_by_recoreds(self, records):
        author_guid_word_embedding_vector = {}
        for record in records:
            author_guid = record[0]
            selected_table_name = record[1]
            selected_targeted_field_name = record[3]
            selected_word_embedding_type = record[4]
            vector = record[5:]
            # vector_str = np.array(vector_str)
            # vector = vector_str.astype(np.float)
            author_guid_word_embedding_vector[author_guid] = vector
        return author_guid_word_embedding_vector

    def get_word_embedding_dictionary(self):
        query = """SELECT * FROM wikipedia_model_300d"""
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        records = list(tuples)
        ans = {record[0]: record[1:301] for record in records}
        return ans

    def get_author_word_embedding_table(self):
        query = """SELECT * FROM author_word_embeddings"""
        query = self.session.execute(query)
        results = pd.read_sql_table('author_word_embeddings', self.engine)
        return results

    def get_author_word_embedding(self, author_guid, table_name, target_field_name, author_word_embeddings):
        ans = {}
        columns = self._get_word_embeddings_types(author_word_embeddings)
        ans = {str(col): self.get_author_guid_word_embedding_vector_dict(author_word_embeddings, table_name,
                                                                             target_field_name, col)[author_guid] for
               col in columns}
        # ans[u'min'] = self.get_author_guid_word_embedding_vector_dict(table_name, target_field_name, u'min')[author_guid]
        # ans[u'max'] = self.get_author_guid_word_embedding_vector_dict(table_name, target_field_name, u'max')[author_guid]
        # ans[u'np.mean'] = self.get_author_guid_word_embedding_vector_dict(table_name, target_field_name, u'np.mean')[author_guid]
        return ans

    def get_records_by_id_targeted_field_and_table_name(self, id_field, targeted_field_name, table_name, where_clauses):
        query = """
                SELECT {0}, {1}
                FROM {2}
                """.format(id_field, targeted_field_name, table_name)

        is_first_condition = False
        for where_clause_dict in where_clauses:
            field_name = where_clause_dict['field_name']
            value = where_clause_dict['value']
            if is_first_condition == False:
                condition_clause = """
                                    WHERE {0} = {1}
                                    """.format(field_name, value)
                is_first_condition = True
            else:
                condition_clause = """
                                    AND {0} = {1}
                                    """.format(field_name, value)
            query += condition_clause
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        return tuples

    def get_word_vector_dictionary(self, table_name):
        query = """
                SELECT *
                FROM {0}
                """.format(table_name)
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        tuples = self.result_iter(cursor)

        word_vector_dict = defaultdict()
        for tuple in tuples:
            word = tuple[0]
            vector = tuple[1:]
            word_vector_dict[word] = vector
        return word_vector_dict

    def fill_author_type_by_post_type(self):
        logging.info("fill_author_type_by_post_type")
        query = 'UPDATE authors SET author_type = (SELECT post_type FROM posts where posts.author_guid = authors.author_guid)'
        query = text(query)
        try:
            self.session.execute(query)
        except Exception as exc:
            logging.error("Fillin author type by post type failed")
        finally:
            self.session.commit()

    def get_item_by_targeted_fields_dict_and_id(self, targeted_fields_dict, id_val):
        query = "SELECT * FROM " + targeted_fields_dict['table_name'] + " where " + targeted_fields_dict[
            'id_field'] + " = '" + id_val + "'"
        result = self.session.execute(query)
        cursor = result.cursor

        result = cursor.fetchall()[0]
        return result

    def get_dict_idfield_to_item(self, targeted_fields_dict):
        id_field = targeted_fields_dict['id_field']
        query = 'select * from ' + targeted_fields_dict['table_name']
        answer = self.session.execute(text(query))
        return dict((getattr(item, id_field), item) for item in self.result_iter(answer))

    def get_author_id_by_field_id(self, field_id, id_val):
        if field_id == "post_id":
            query = 'SELECT author_guid FROM posts WHERE post_id=' + id_val
            answer = self.session.execute(text(query))
            cursor = answer.cursor
            result = cursor.fetchall()[0]
            return result[0]
        if field_id == "author_guid":
            return id_val

    def get_liar_dataset_records(self):
        liar_dataset_records = self.session.query(Politifact_Liar_Dataset).all()
        return liar_dataset_records

    def randomize_authors(self, min_number_of_posts_per_author, domain, authors_table_field_name,
                          authors_table_value, num_of_random_authors):
        randomized_authors = []
        randomized_authors_for_graph = self._randomize_authors_by_conditions(min_number_of_posts_per_author,
                                                                             domain, authors_table_field_name,
                                                                             authors_table_value,
                                                                             num_of_random_authors)
        for author_guid, author_type in randomized_authors_for_graph:
            randomized_author_for_graph = self._create_randomized_author_for_graph(author_guid, author_type)
            randomized_authors.append(randomized_author_for_graph)

        self.addPosts(randomized_authors)

    def _randomize_authors_by_conditions(self, min_posts_count, domain, authors_table_field_name,
                                         authors_table_value, num_of_random_authors):

        query = """
                SELECT authors.author_guid, authors.author_type
                FROM authors
                INNER JOIN author_guid_num_of_posts_view ON (author_guid_num_of_posts_view.author_guid = authors.author_guid)
                WHERE author_guid_num_of_posts_view.num_of_posts >= {0}
                AND authors.domain = '{1}'
                AND authors.{2} = '{3}'
                ORDER BY RANDOM()
                LIMIT {4}
                """.format(min_posts_count, domain, authors_table_field_name, authors_table_value,
                           num_of_random_authors)
        query = text(query)
        result = self.session.execute(query, params=dict(min_posts_count=min_posts_count, domain=domain,
                                                         num_of_random_authors=num_of_random_authors))
        cursor = result.cursor
        randomized_authors_for_graph = self.result_iter(cursor)
        return randomized_authors_for_graph

    def get_author_screen_name_author_guid_dictionary(self):
        query = """
                SELECT authors.author_screen_name, authors.author_guid
                FROM authors
                """
        query = text(query)

        result = self.session.execute(query, params=dict(domain=domain))
        cursor = result.cursor
        tuples = self.result_iter(cursor)
        author_screen_name_author_guid_dict = {}
        for tuple in tuples:
            author_screen_name = tuple[0]
            author_guid = tuple[1]
            author_screen_name_author_guid_dict[author_screen_name] = author_guid
        return author_screen_name_author_guid_dict

    def get_resturant_api_id_to_resturant_dict(self):
        authors = self.get_all_authors()
        authors_dict = dict((str(aut.author_guid).encode('utf-8'), aut) for aut in authors)
        return authors_dict

    def fix_guids_encoding(self):
        guids = self.get_author_guids()
        for guid in guids:
            new_guid = str(guid).encode('ascii', 'ignore').decode('ascii')
            update_query = "UPDATE " + self.authors + " SET author_guid ='" + new_guid + "' WHERE author_guid ='" + guid + "'"
            self.update_query(update_query)

    def get_politifact_speaker_by_author(self, author):
        query = text("SELECT party_affiliation FROM politifact_liar_dataset where post_guid ='" + author + "'")
        result = self.session.execute(query)
        cursor = result.cursor
        records = list(cursor.fetchall())[0]
        return records

    def export_word_embeddings_to_tsv(self):
        embeddings_dict = self.get_author_guid_word_embedding_vector_dict('posts', 'content', 'max')
        file = open("D:\\Work\\BadActorsFolder\\trunk\\software\\bad_actors\\data\\output\\word_embeddings_mean2.tsv",
                    'wb')
        file_labled = open(
            "D:\\Work\\BadActorsFolder\\trunk\\software\\bad_actors\\data\\output\\word_embeddings_mean_lables.tsv",
            'wb')
        writer = csv.writer(file, delimiter='\t')
        counter = 0
        for author in embeddings_dict:
            record = embeddings_dict[author]
            counter += 1
            str_record = str(record)[1:-1]
            str_record = str_record.replace(', ', '\t')
            file.write(str_record + '\n')

            author_type = (self.get_author_by_guid(author)).author_type
            author = self.get_politifact_speaker_by_author(author)[0]

            file_labled.write(author + '\t' + author_type + '\n')
        file.close()
        file_labled.close()

    ##
    ## Added for solving the issue of US Elections
    ##

    def convert_tweets_to_posts_and_authors(self, tweets, domain):
        posts = []
        authors = []
        for tweet in tweets:
            post, author = self._convert_tweet_to_post_and_author(tweet, domain)
            posts.append(post)
            authors.append(author)

        posts = list(set(posts))
        authors = list(set(authors))
        return posts, authors

    def _convert_tweet_to_post_and_author(self, tweet, domain):
        post = self._convert_tweet_to_post(tweet, domain)
        author = self._convert_tweet_to_author(tweet, domain)

        return post, author

    def convert_tweet_to_user_mentions(self, tweet, post_guid):
        tweet_user_mentions = tweet.user_mentions
        user_mentions = []
        for tweet_user_mention in tweet_user_mentions:
            user_mention = PostUserMention()

            user_mention.post_guid = post_guid
            user_mention.user_mention_twitter_id = tweet_user_mention.id_str
            user_mention.user_mention_screen_name = tweet_user_mention.screen_name

            user_mentions.append(user_mention)
        return user_mentions

    def _convert_tweet_to_post(self, tweet, domain):
        post = Post()

        tweet_id = tweet.id_str
        post.post_osn_id = tweet_id
        post.retweet_count = tweet.retweet_count
        post.favorite_count = tweet.favorite_count
        post.content = tweet.text

        user = tweet.user
        screen_name = user.screen_name
        post.author = screen_name

        url = "https://twitter.com/{0}/status/{1}".format(screen_name, tweet_id)
        post.url = url

        created_at = tweet.created_at
        post.created_at = created_at
        tweet_str_publication_date = str(extract_tweet_publiction_date(created_at))
        tweet_creation_date = str_to_date(tweet_str_publication_date)
        post.date = tweet_creation_date
        post.domain = domain

        post_guid = compute_post_guid(url, screen_name, tweet_str_publication_date)
        post.post_id = post_guid
        post.guid = post_guid

        author_guid = compute_author_guid_by_author_name(screen_name)
        post.author_guid = author_guid
        post.post_format = tweet.lang
        return post

    def _convert_tweet_to_author(self, tweet, domain):

        author = Author()

        user = tweet.user
        screen_name = user.screen_name
        author_guid = compute_author_guid_by_author_name(screen_name)

        author.author_guid = author_guid
        author.name = screen_name
        author.author_screen_name = screen_name
        author.created_at = user.created_at
        author.description = user.description
        author.favourites_count = user.favourites_count
        author.followers_count = user.followers_count
        author.friends_count = user.friends_count
        author.statuses_count = user.statuses_count

        author.geo_enabled = user.geo_enabled

        user_id = user.id_str
        author.author_osn_id = user_id
        author.language = user.lang
        author.listed_count = user.listed_count
        author.profile_background_color = user.profile_background_color
        author.profile_image_url = user.profile_background_image_url
        author.profile_background_tile = user.profile_background_tile
        author.profile_banner_url = user.profile_banner_url
        author.profile_link_color = user.profile_link_color

        author.profile_sidebar_fill_color = user.profile_sidebar_fill_color
        author.profile_text_color = user.profile_text_color
        author.location = user.location if user.location != None else None
        author.protected = user.protected if user.protected != None else None
        author.time_zone = user.time_zone if user.time_zone != None else None
        author.url = user.url if user.url != None else None
        author.utc_offset = user.utc_offset if user.utc_offset != None else None
        author.domain = domain
        author.verified = user.verified

        return author

    def add_claim_connections(self, claim_connections):
        i = 1
        for claim in claim_connections:
            if (i % 100 == 0):
                msg = "\r Insert claim_connection to DB: [{}".format(i) + "/" + str(len(claim_connections)) + ']'
                print(msg, end="")
            i += 1
            self.session.merge(claim)
        msg = "\r Insert claim_connection to DB: [{}".format(i) + "/" + str(len(claim_connections)) + ']'
        print(msg)
        self.commit()

    def get_claim_tweet_connections(self):
        q = """
            SELECT *
            FROM claim_tweet_connection            
            """
        query = text(q)
        result = self.session.execute(query)
        return list(result)

    def get_claim_ordered_by_num_of_posts(self):
        q = """
            SELECT claim_tweet_connection.claim_id, COUNT(claim_tweet_connection.post_id)
            FROM claim_tweet_connection           
            GROUP BY 1
            ORDER BY 2 DESC 
            """
        query = text(q)
        result = self.session.execute(query)
        return list(result)

    def get_table_by_name(self, table_name):
        for var in globals():
            class_obj = globals()[var]
            try:
                if issubclass(class_obj, Base) and class_obj.__tablename__ == table_name:
                    return class_obj
            except:
                pass
        return None

    def get_verified_authors(self):
        results = self.session.query(Author).filter(Author.verified == '1').all()
        return results

    def get_post_osn_ids(self):
        q = """
            SELECT post_osn_id
            FROM posts
            """
        query = text(q)
        result = self.session.execute(query)
        return list(result)

    def get_posts_by_selected_domain(self, domain):
        q = """
            SELECT posts.post_id, posts.post_type
            FROM posts
            WHERE posts.domain = :domain
            """
        query = text(q)
        result = self.session.execute(query, params=dict(domain=domain))
        return list(result)

    def get_claim_id_posts_dict(self):
        claim_id_posts_dict = defaultdict(list)
        post_dict = self.get_post_dictionary()
        for claim_id, post_id in self.get_claim_tweet_connections():
            claim_id_posts_dict[claim_id].append(post_dict[post_id])
        return claim_id_posts_dict

    def get_claim_posts(self, limit = 1):
        # table_elements = self.session.query(connection_source_attr, destination_table) \
        #     .join(source_table, connection_source_attr == source_id_attr) \
        #     .join(destination_table, connection_target_attr == destination_id_attr) \
        #     .filter(and_(condition for condition in conditions)) \
        #     .order_by(connection_source_attr) \
        #     .yield_per(10000).enable_eagerloads(False).offset(offset)


        claim_posts = self.session.query(Claim_Tweet_Connection.claim_id, Post)\
            .join(Claim_Tweet_Connection, Post.post_id == Claim_Tweet_Connection.post_id)\
            .order_by(Claim_Tweet_Connection.claim_id).yield_per(10000).enable_eagerloads(False)
        claim_id_posts = defaultdict(list)
        for i, (claim_id, post) in enumerate(claim_posts):
            if claim_id not in claim_id_posts and len(claim_id_posts) == limit:
                yield claim_id_posts
                claim_id_posts = defaultdict(list)

            claim_id_posts[claim_id].append(post)
        # print()
        yield claim_id_posts

    def get_author_friends_by_sources(self, author_guids):
        author_connections = self.session.query(AuthorConnection.source_author_guid, AuthorConnection.destination_author_guid) \
            .filter(and_(or_(AuthorConnection.source_author_guid.in_(author_guids), AuthorConnection.destination_author_guid.in_(author_guids)), AuthorConnection.connection_type == 'friend')).yield_per(10000).enable_eagerloads(False)

        author_friends = defaultdict(set)
        for i, (source_author, dest_author) in enumerate(author_connections):
            author_friends[source_author].add(dest_author)
            author_friends[dest_author].add(source_author)
        return author_friends

    def get_claim_post_author_connections_with_verdict(self):
        q = """
            SELECT claim_tweet_connection.claim_id, claim_tweet_connection.post_id, posts.author_guid, claims.verdict
            FROM claim_tweet_connection
            INNER JOIN posts ON (posts.post_id  =  claim_tweet_connection.post_id)
            INNER JOIN claims ON (claims.claim_id  =  claim_tweet_connection.claim_id)
            """
        query = text(q)
        result = self.session.execute(query)
        return list(result)

    def get_claim_post_author_connections(self):
        q = """
            SELECT claim_tweet_connection.claim_id, claim_tweet_connection.post_id, posts.author_guid
            FROM claim_tweet_connection
            INNER JOIN posts ON (posts.post_id = claim_tweet_connection.post_id)
            """
        query = text(q)
        result = self.session.execute(query)
        return list(result)

    def delete_author_sub_type(self):
        query = '''
                UPDATE authors
                SET author.author_sub_type = NULL
                '''
        self.update_query(query)

    def add_claim_keywords_connection(self, claim_id, type, keywords):
        claim_keywords_connections = Claim_Keywords_Connections()
        claim_keywords_connections.claim_id = claim_id
        claim_keywords_connections.keywords = keywords
        claim_keywords_connections.type = type
        self.addPost(claim_keywords_connections)

    def add_claim_keywords_connections(self, connections):
        self.addPosts(connections)

    def get_claim_keywords_connections(self):
        return self.session.query(Claim_Keywords_Connections).all()

    def get_claim_keywords_connections_by_type(self, type):
        return self.session.query(Claim_Keywords_Connections).filter(Claim_Keywords_Connections.type == type).all()

    def get_claim_id_keywords_dict_by_connection_type(self, type):
        connections = self.get_claim_keywords_connections_by_type(type)
        return {connection.claim_id: connection.keywords for connection in connections}

    def add_claim_tweet_connections_fast(self, claim_post_connections):
        table_name = 'claim_tweet_connections'
        self.add_entity_fast(table_name, claim_post_connections)

    def add_posts_fast(self, posts):
        table_name = 'posts'
        keys = ['post_id', 'domain']
        self.add_entity_fast(table_name, posts)

    def add_terms_fast(self, terms):
        table_name = 'terms'
        keys = ['term_id']
        self.add_entity_fast(table_name, terms)

    def add_claim_keywords_connections(self, claim_keywords):
        self.add_entity_fast('claim_keywords', claim_keywords)

    def add_topic_items_fast(self, topic_items):
        table_name = 'topics'
        keys = ['topic_id', 'term_id']
        self.add_entity_fast(table_name, topic_items)

    def add_post_topic_mappings_fast(self, post_topic_mappings):
        table_name = 'post_topic_mapping'
        keys = ['post_id']
        self.add_entity_fast(table_name, post_topic_mappings)

    def add_news_articles_fast(self, news_articles):
        table_name = 'news_articles'
        keys = ['article_id']
        self.add_entity_fast(table_name, news_articles)

    def add_author_connections_fast(self, author_connections):
        keys = ['source_author_guid', 'destination_author_guid', 'connection_type']
        table_name = 'author_connections'
        self.add_entity_fast(table_name, author_connections)

    def add_author_features_fast(self, author_features):
        keys = ['author_guid', 'window_start', 'window_end', 'attribute_name']
        table_name = 'author_features'
        self.insert_or_update(author_features)

    def add_reddit_authors(self, authors):
        keys = ['name', 'author_guid']
        table_name = 'reddit_authors'
        self.add_entity_fast(table_name, authors)

    def add_claims_fast(self, claims):
        keys = ['claim_id']
        table_name = 'claims'
        self.add_entity_fast(table_name, claims)

    def add_authors_fast(self, authors):
        keys = ['author_guid', 'name', 'domain']
        table_name = 'authors'
        self.add_entity_fast(table_name, authors)

    def get_entity_key(self, table_item):
        return self.inspect_item(table_item).identity

    def inspect_item(self, table_item):
        return inspect(table_item)

    def map_item(self, item, table_mapper):
        return mapper(item, table_mapper)

    def add_entity_fast(self, table_name, entities):
        try:
            self.session.bulk_save_objects(entities)
            self.session.commit()
            print('Add {} new {} to db'.format(len(entities), table_name))
            return
        except Exception as e:
            count = len(entities)
            self.session.rollback()
            if count <= 10000:
                self.merge_items(entities)
                self.session.commit()
            else:
                self.add_entity_fast(table_name, entities[:(count/2)])
                self.add_entity_fast(table_name, entities[(count/2):])

    # def add_authors_fast(self, authors):
    #     self.add_items_fast(self.get_author_dictionary(), authors, 'author_guid', 'authors')

    # def add_author_features_fast(self, author_features):
    #     author_feature_dict = {author_feature.author_guid: author_feature for author_feature in self.get_author_features()}
    #     self.add_items_fast(author_feature_dict, author_features, 'author_guid', 'author_features')

    def add_items_fast(self, item_dict, items, item_key, item_type='items'):
        self.session.close()
        filtered_items = []
        for item in items:
            if getattr(item, item_key) not in item_dict:
                filtered_items.append(item)
                item_dict[getattr(item, item_key)] = item
        duplication_count = str(len(items) - len(filtered_items))
        print('Add {} new {} to db, remove {} duplications'.format(len(filtered_items), item_type, duplication_count))
        self.session.bulk_save_objects(filtered_items)
        self.session.commit()

    def get_claims_tuples(self):
        q = """
            SELECT *
            FROM claims
            """
        query = text(q)
        result = self.session.execute(query)
        return list(result)

    def get_posts_from_domain_contain_words(self, domain, words):
        conditions = [Post.content.like('%{}%'.format(word)) for word in words]
        return self.session.query(Post).filter(and_(Post.domain == domain, and_(*conditions))).all()

    def insert_or_update(self, items):
        try:
            self.session.bulk_save_objects(items, update_changed_only=False)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            self.addPosts(items)

    def delete_class_author_features(self, class_name):
        self.session.query(AuthorFeatures).filter(AuthorFeatures.attribute_name.startswith(class_name)).delete(
            synchronize_session=False)
        self.session.commit()

    def drop_table(self, table_name):
        if self.is_table_exist(table_name):
            table = self.get_table_by_name(table_name)
            table.__table__.drop(self.engine)

    def get_authors_with_feature(self, feature_prefix_name, author_guids=None):
        if not author_guids:
            result = self.session.query(AuthorFeatures.author_guid).distinct(AuthorFeatures.author_guid) \
                .filter(AuthorFeatures.attribute_name.startswith(feature_prefix_name)).all()
        else:
            result = self.session.query(AuthorFeatures.author_guid).distinct(AuthorFeatures.author_guid) \
                .filter(and_(AuthorFeatures.attribute_name.startswith(feature_prefix_name),
                             AuthorFeatures.author_guid.in_(set(author_guids)))).all()
        return [r[0] for r in result]

    def get_authors_with_enough_features(self, feature_prefix_name, count):
        result = self.session.query(AuthorFeatures.author_guid, func.count(AuthorFeatures.attribute_name)) \
            .filter(AuthorFeatures.attribute_name.startswith(feature_prefix_name)) \
            .group_by(AuthorFeatures.author_guid).having(func.count(AuthorFeatures.attribute_name) == count).all()
        return [r[0] for r in result]

    def get_authors_with_features(self, feature_names, author_guids=None):
        # author_sets = [set(self.get_authors_with_feature(feature_name, author_guids)) for feature_name in feature_names]
        result = self.session.query(AuthorFeatures.author_guid).distinct(AuthorFeatures.author_guid) \
            .filter(and_(AuthorFeatures.attribute_name.startswith(feature_name) for feature_name in feature_names))
        return set.intersection(*author_sets)

    def get_hospital_twitter_screen_names(self):
        q = """
            SELECT hospital_tweet_users_with_screen_names.screen_name
            FROM hospital_tweet_users_with_screen_names
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]
        # return screen_names
        #return list(result)

    def get_labor_employees_screen_names(self):
        q = """
            SELECT labor_unions_tweet_users_with_screen_names.screen_name
            FROM labor_unions_tweet_users_with_screen_names
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]

    def get_follower_ids_of_hospitals_and_labor_unions(self):
        q = """
            SELECT union_healthcare_and_labor_union_workers.destination_author_osn_id
            FROM union_healthcare_and_labor_union_workers
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        ids = self.result_iter(cursor)
        return [id[0] for id in ids]


    def get_intersection_of_labor_union_and_healthcare_users_followers_ids(self):
        q = """
            SELECT DISTINCT(temp_author_connections.destination_author_osn_id)
            FROM temp_author_connections
            WHERE temp_author_connections.destination_author_osn_id IN (
                SELECT healtchcare_temp_author_connections.destination_author_osn_id
                FROM healtchcare_temp_author_connections
            )
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        ids = self.result_iter(cursor)
        return [id[0] for id in ids]


    #
    # collect all users who we finished to collect their followers_ids
    def get_already_retrieved_their_follower_ids(self):
        q = """
            SELECT DISTINCT(temp_author_connections.source_author_osn_id)
            FROM temp_author_connections
            WHERE temp_author_connections.connection_type = 'follower'
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]


        # SELECT DISTINCT(temp_author_connections.destination_author_osn_id)
        # FROM temp_author_connections
        # WHERE temp_author_connections.source_author_osn_id NOT IN (
        #     SELECT labor_unions_users_with_screen_names_osn_ids_and_relevance.author_osn_id
        #     FROM labor_unions_users_with_screen_names_osn_ids_and_relevance
        #     WHERE labor_unions_users_with_screen_names_osn_ids_and_relevance.is_related_to_healthcare_using_Wikipedia = '0'
        # )
        # AND temp_author_connections.destination_author_osn_id NOT IN (
        #     SELECT labor_unions_users_with_screen_names_osn_ids_and_relevance.author_osn_id
        #     FROM labor_unions_users_with_screen_names_osn_ids_and_relevance
        # )

    def get_healthcare_labor_union_follower_ids(self):
        q = """
            SELECT DISTINCT(temp_author_connections.destination_author_osn_id)
            FROM temp_author_connections
            WHERE temp_author_connections.destination_author_osn_id NOT IN (
                SELECT labor_unions_users_with_screen_names_osn_ids_and_relevance.author_osn_id
                FROM labor_unions_users_with_screen_names_osn_ids_and_relevance
            )
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        osn_ids = self.result_iter(cursor)
        return [r[0] for r in osn_ids]


    def get_poi_screen_names(self):
        q = """
            SELECT Optional_POIs_with_twitter_screen_name.twitter_author_screen_name
            FROM Optional_POIs_with_twitter_screen_name
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]

    def get_ukraine_russia_conflict_poi_screen_names(self):
        q = """
            SELECT ukraine_russia_conflict_pois.user_screen_name
            FROM ukraine_russia_conflict_pois
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]

    def get_poi_v6_screen_names(self):
        q = """
            SELECT Health_POIs_V6_with_screen_names.author_screen_name
            FROM Health_POIs_V6_with_screen_names
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]

    def get_follower_ids_to_crawl(self):
        q = """
            SELECT DISTINCT(temp_author_connections.destination_author_osn_id)
            FROM temp_author_connections
            WHERE temp_author_connections.destination_author_osn_id NOT IN (
                SELECT authors.author_osn_id
                FROM authors
            )
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        author_ids = self.result_iter(cursor)
        return [r[0] for r in author_ids]

    def get_author_ids_not_general_public_and_not_brought_followers_for_them(self):
        q = """
            SELECT Health_POIs_V6_with_screen_names_and_osn_ids.author_osn_id
            FROM Health_POIs_V6_with_screen_names_and_osn_ids
            WHERE Health_POIs_V6_with_screen_names_and_osn_ids.general_public_interest = ''
            AND Health_POIs_V6_with_screen_names_and_osn_ids.author_osn_id NOT IN (
                SELECT DISTINCT(temp_author_connections.source_author_osn_id)
                FROM temp_author_connections
            )
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        author_ids = self.result_iter(cursor)
        return [r[0] for r in author_ids]

    def get_description_and_full_names_for_authors(self):
        q = """
            SELECT authors.author_guid, authors.author_osn_id, authors.author_screen_name, authors.author_full_name, authors.description
            FROM authors
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        records = self.result_iter(cursor)
        return records


    def get_healthcare_worker_screen_names(self):
        q = """
            SELECT authors.author_screen_name
            FROM authors
            WHERE authors.author_type = 'Healthcare_Worker_Auto'
            """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        screen_names = self.result_iter(cursor)
        return [r[0] for r in screen_names]

    def get_spokesmanships_screen_names(self):
        q = """
               SELECT spokesmanships.Twitter_Screen_Name
               FROM spokesmanships
               WHERE spokesmanships.Twitter_Screen_Name IS NOT NULL
              """
        query = text(q)
        result = self.session.execute(query)
        cursor = result.cursor
        author_screen_names = self.result_iter(cursor)
        return [r[0] for r in author_screen_names]

        return set.intersection(*author_sets)

    def get_authors_by_popularity(self):
        query = """
                SELECT posts.author_guid, count(posts.post_id)
                FROM posts
                GROUP BY 1
                ORDER BY 2 Desc
                """
        # posts = self.session.query(Post).filter(Post.domain == unicode(domain)).slice(start,stop).all()
        query = text(query)
        result = self.session.execute(query)
        cursor = result.cursor
        author_popularity_tupls = self.result_iter(cursor)  # , a
        return author_popularity_tupls