
import json
from DB.schema_definition import AuthorConnection
from commons.commons import clean_tweet, clean_content_by_nltk_stopwords
from dataset_builder.feature_extractor.feature_argument_parser import ArgumentParser


class EntityToTopicConverter(ArgumentParser):
    def __init__(self, db):
        super(EntityToTopicConverter, self).__init__(db)
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._output_path = self._config_parser.eval(self.__class__.__name__, "json_output_path")
        self._remove_stop_words = self._config_parser.eval(self.__class__.__name__, "remove_stop_words")
        self.term_dictionary = {}
        self._last_term_index = 0
        self._num_topics = 0
        self._source_id_topic_dict = {}
        self._entity_id_to_topic = {}

        self._term_id_term_texts = []
        self._topic_id_term_ids = []
        self._post_topic_mappings = []
        self._author_mappings = []
        self._author_connections = []

    def execute(self, window_start=None):
        for arg in self._targeted_fields:
            self._num_topics = 0
            print("load data from DB")
            source_id_target_elements_dict = self._get_source_id_target_elements(arg)
            print("Generate topic table")
            self.generate_topics_tables(source_id_target_elements_dict, arg)
            print("Generate post_topic_mapping table")
            self.generate_post_topic_mapping(source_id_target_elements_dict, arg)
            print("Generate author_topic_mapping table")
            self.generate_author_topic_mapping()
            print("Add source_id topic_id to author_connections")
            self._add_source_id_topic_id_to_author_connections(arg)
            print("Write source_id_topic_id_dict to json")
            self._write_source_id_topic_id_to_json(arg)

            self.save_topic_entities()

    def save_topic_entities(self):
        self._db.add_terms_fast(self._term_id_term_texts)
        self._db.add_topic_items_fast(self._topic_id_term_ids)
        self._db.add_post_topic_mappings_fast(self._post_topic_mappings)
        self._db.add_author_connections_fast(self._author_connections)
        self._db.insert_into_author_toppic_mappings(self._author_mappings)
        self._term_id_term_texts = []
        self._topic_id_term_ids = []
        self._post_topic_mappings = []
        self._author_mappings = []
        self._author_connections = []
        self._db.session.close()

    def _write_source_id_topic_id_to_json(self, arg):
        table_name = arg['source']['table_name']
        id_name = arg['source']['id']
        json_str = json.dumps(self._source_id_topic_dict, ensure_ascii=False)
        json_file = open('{2}{0}_{1}_topic_id_json.json'.format(table_name, id_name, self._output_path), 'w')
        json_file.write(json_str)

    def _add_source_id_topic_id_to_author_connections(self, arg):
        table_name = arg['source']['table_name']
        id_name = arg['source']['id']
        connection_type = '{0}.{1} to topic id'.format(table_name, id_name)
        author_connections = []
        for source_id, topic_id in self._source_id_topic_dict.items():
            ac = AuthorConnection()
            ac.source_author_guid = source_id
            ac.destination_author_guid = topic_id
            ac.connection_type = connection_type
            author_connections.append(ac)
        self._author_connections = author_connections
        # self._db.add_author_connections(author_connections)

    def get_source_id_topic_dictionary(self):
        return self._source_id_topic_dict

    def generate_topics_tables(self, source_id_target_elements_dict, target_fields):
        # source_id_target_elements_dict = self._get_source_id_target_elements(target_fields)
        table_name = target_fields['source']['table_name']
        id_name = target_fields['source']['id']
        field_name = target_fields['source']['target_field']
        where_clauses = target_fields['source']['where_clauses']

        source_ids = list(source_id_target_elements_dict.keys())
        topic_objects = self._db.get_table_elements_by_ids(table_name, id_name, source_ids, where_clauses)
        term_id_term_texts = []
        topic_id_term_ids = []
        i = 0
        for topic_obj in topic_objects:
            if i % 100 == 0:
                msg = "\r generate topic {0}/{1}".format(str(i), str(len(topic_objects)))
                print(msg, end="")
            i += 1
            source_id = getattr(topic_obj, id_name)
            topic_content = clean_tweet(getattr(topic_obj, field_name))
            if self._remove_stop_words:
                topic_content = clean_content_by_nltk_stopwords(topic_content)
            topic_terms = topic_content.split(' ')
            for term in topic_terms:
                if term not in self.term_dictionary:
                    self.term_dictionary[term] = self._last_term_index
                    self._last_term_index += 1
            term_count = float(len(topic_terms))
            self._num_topics += 1
            term_id_term_txt = [self._db.create_term(self.term_dictionary[term], term) for term in topic_terms]
            topic_id_term_id = [
                self._db.create_topic_item(self._num_topics, self.term_dictionary[term],
                                           topic_content.count(term) / term_count) for
                term in topic_terms]
            self._source_id_topic_dict[source_id] = self._num_topics
            term_id_term_texts.extend(term_id_term_txt)
            topic_id_term_ids.extend(topic_id_term_id)
        msg = "\r generate topic {0}/{1}".format(str(i), str(len(topic_objects)))
        print(msg, end="")
        print()
        self._term_id_term_texts = term_id_term_texts
        self._topic_id_term_ids = topic_id_term_ids
        # self._db.add_terms(term_id_term_texts)
        # self._db.add_topic_items(topic_id_term_ids)

    def generate_post_topic_mapping(self, source_id_target_elements_dict, target_fields):
        # source_id_target_elements_dict = self._get_source_id_target_elements(target_fields)
        destination_id_field = self._get_arg_value(target_fields, 'id')
        post_topic_mappings = []
        i = 0
        for source_id, target_elements in source_id_target_elements_dict.items():
            for element in target_elements:
                if i % 100 == 0:
                    msg = "\r generate post mapping {0}".format(str(i))
                    print(msg, end="")
                i += 1
                element_id = getattr(element, destination_id_field)
                max_topic_probability = (self._source_id_topic_dict[source_id], 1)
                ptm = self._db.create_post_topic_mapping_obj(max_topic_probability, element_id)
                post_topic_mappings.append(ptm)
                self._entity_id_to_topic[element_id] = self._source_id_topic_dict[source_id]
        msg = "\r generate post mapping {0}".format(str(i))
        print(msg, end="")
        print()
        self._post_topic_mappings = post_topic_mappings
        # self._db.addPostTopicMappings(post_topic_mappings)
        pass

    def _get_arg_value(self, target_fields, arg_name):
        if self._contain_destination_arg(target_fields):
            arg_value = target_fields['destination'][arg_name]
        else:
            arg_value = target_fields['source'][arg_name]
        return arg_value

    def _contain_destination_arg(self, target_fields):
        return "destination" in target_fields and target_fields["destination"] != {}

    def generate_author_topic_mapping(self):
        post_to_topic_id = {source_id: {topic_id: 1.0} for source_id, topic_id in self._entity_id_to_topic.items()}
        self.create_author_topic_mapping_table(post_to_topic_id)

    def create_author_topic_mapping_table(self, post_to_topic_id):
        authors = self._db.get_authors_by_domain(self._domain)
        self._db.delete_author_topic_mapping()
        self._db.create_author_topic_mapping_table(self._num_topics)
        author_guid_posts_dict = self._db.get_author_guid_post_dict()
        author_mappings = []
        i = 0
        for author in authors:
            if i % 100 == 0:
                msg = "\r generate author mapping {0}/{1}".format(str(i), str(len(authors)))
                print(msg, end="")
            i += 1
            posts_by_domain = author_guid_posts_dict[author.author_guid]
            posts_by_domain = [post for post in posts_by_domain if post.domain == self._domain]
            post_count = len(posts_by_domain)
            topics_probabilities = [0] * self._num_topics
            for post in posts_by_domain:
                if post.guid in post_to_topic_id:
                    for key in post_to_topic_id[post.guid]:
                        topics_probabilities[key - 1] += 1.0 / post_count
            author_mappings.append((str(author.author_guid), topics_probabilities))
        msg = "\r generate author mapping {0}/{1}".format(str(i), str(len(authors)))
        print(msg, end="")
        print()
        self._author_mappings = author_mappings
        # self._db.insert_into_author_toppic_mappings(author_mappings)
