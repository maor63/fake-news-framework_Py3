

from commons.commons import convert_claim_to_post
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.fake_news_word_classifier.fake_news_base import FakeNewsBase


class FakeNewsFeatureGenerator(FakeNewsBase):
    def __init__(self, db):
        super(FakeNewsFeatureGenerator, self).__init__(db, **{'authors': {}, 'posts': []})

    def extract_word_count(self):
        print("extract word_count features")
        self._extract_word_feature_base(self._calc_source_id_word_count_dict, "count")

    def extract_word_fraction(self):
        print("extract word_fraction features")
        self._extract_word_feature_base(self._calc_source_id_word_fraction_dict, "fraction")

    def extract_claim_verdict(self):
        print("extract claim_type features")
        claims = self._db.get_claims()
        authors_features = []
        win_start = self._window_start
        win_end = self._window_end
        for claim in claims:
            attr_name = self._prefix + "_claim_verdict"
            verdict = self._get_claim_verdict(claim)
            if verdict != "":
                author_feature = BaseFeatureGenerator.create_author_feature(attr_name, claim.claim_id, verdict,
                                                                            win_start,
                                                                            win_end)
                authors_features.append(author_feature)
                if len(authors_features) >= 10000:
                    self._db.add_author_features_fast(authors_features)
                    authors_features = []
        self._db.add_author_features_fast(authors_features)

    def _extract_word_feature_base(self, calc_source_id_word_data_dict_fn, featue_type):
        authors_features = []
        for targeted_fields_dict in self._targeted_fields:
            print("Get sourse id target dict")
            for source_id_target_elements_dict in self._get_source_id_target_elements_generator(targeted_fields_dict):
                source_ids = list(source_id_target_elements_dict.keys())
                source_id_source_element_dict = self._get_source_id_source_element_dict(source_ids,
                                                                                        targeted_fields_dict)
                for source_targets_dict_item in source_id_target_elements_dict.items():
                    source_id = source_targets_dict_item[0]
                    author = source_id_source_element_dict[source_id]
                    kwargs = self._get_feature_kwargs(source_targets_dict_item, author, targeted_fields_dict)
                    _claim_id_word_data_dict = calc_source_id_word_data_dict_fn(kwargs)
                    authors_features += self._extract_features_from_claim_to_word_data(_claim_id_word_data_dict,
                                                                                       featue_type, source_id)
                    if len(authors_features) >= self._max_objects_save:
                        self._db.add_author_features_fast(authors_features)
                        authors_features = []
        self._db.add_author_features_fast(authors_features)
        self._db.session.commit()

    def _get_claim_verdict(self, claim):
        value = ""
        if claim.verdict in self._bad:
            value = 'False'
        elif claim.verdict in self._good:
            value = 'True'
        return value

    def _extract_features_from_claim_to_word_data(self, claim_id_words_frec_dict, type_name, claim_id):
        authors_features = []
        win_start = self._window_start
        win_end = self._window_end
        for word in self._fake_news_dictionary:
            value = claim_id_words_frec_dict[word]
            attr_name = "{0}_{1}_{2}".format(self._prefix, word.strip().replace(' ', '-'), type_name)
            author_feature = BaseFeatureGenerator.create_author_feature(attr_name, claim_id, value, win_start,
                                                                        win_end)
            authors_features.append(author_feature)
        attr_name = "{0}_words_{1}_sum".format(self._prefix, type_name)
        value = sum(claim_id_words_frec_dict.values())
        author_feature = BaseFeatureGenerator.create_author_feature(attr_name, claim_id, value, win_start, win_end)
        authors_features.append(author_feature)
        # self._db.add_author_features(authors_features)
        return authors_features
