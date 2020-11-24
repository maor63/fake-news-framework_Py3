
from preprocessing_tools.abstract_controller import AbstractController
from commons.commons import *
from DB.schema_definition import SinglePostByAuthor
from .base_feature_generator import BaseFeatureGenerator
from DB.schema_definition import AuthorFeatures

class TopicFeatureGenerator(BaseFeatureGenerator):
    def set_up(self):
        pass

    def execute(self):
        start_time = time.time()
        info_msg = "execute started for " + self.__class__.__name__
        logging.info(info_msg)

        post_topics_map = self._db.get_single_post_per_author_topic_mapping()
        total = len(post_topics_map)
        processed = 0
        authors_features = []
        for tuple in post_topics_map:
            author_guid = tuple[0]
            for topic in range(1,len(tuple)):
                probability = tuple[topic]
                author_feature = AuthorFeatures()
                author_feature.author_guid = author_guid
                author_feature.window_start = self._window_start
                author_feature.window_end = self._window_end
                author_feature.attribute_name = str('probability_topic_'+str(topic))
                author_feature.attribute_value = probability
                authors_features.append(author_feature)

            processed += 1
            print ("\r processed authors " + str(processed) + " from " + str(total), end="")

        if authors_features:
            print ('\n Beginning merging author_features objects')
            counter = 0
            for author_features_row in authors_features:
                counter += 1
                self._db.update_author_features(author_features_row)
                if counter == 100:
                    print ("\r " + str(self.__class__.__name__) + " merging author-features objects", end="")
                    self._db.commit()
                    counter = 0
            if counter != 0:
                self._db.commit()
            print ('Finished merging author_features objects')

        end_time = time.time()
        diff_time = end_time - start_time
        print ('execute finished in ' + str(diff_time) + ' seconds')

