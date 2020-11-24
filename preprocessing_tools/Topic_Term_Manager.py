class Topic_Term_Manager():
    def __init__(self, db):
        self._db = db
        self._topic_top_terms_dict = {}

    def get_term_from_db_with_most_posts(self):
        self._db.create_topic_stats_view()
        topic_id_max_posts_tuple = self._db.get_topic_with_maximal_posts()
        topic_id = topic_id_max_posts_tuple[0]
        terms = self._db.get_top_terms_by_topic_id(topic_id)
        return terms

    def get_topic_top_terms_dictionary(self, num_of_top_terms):
        self._topic_top_terms_dict = self._create_topic_top_terms_dictionary(num_of_top_terms)
        return self._topic_top_terms_dict

    def _create_topic_top_terms_dictionary(self, num_of_top_terms):
        topic_top_terms_dict = {}

        max_topic = self._db.get_max_topic()
        for i in range(1, max_topic):
            top_terms = self._db._get_top_terms_by_topic_id(i, num_of_top_terms)
            topic_top_terms_dict[i] = top_terms
        return topic_top_terms_dict


