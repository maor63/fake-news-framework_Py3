# Created by aviade
# Time: 31/03/2016 14:22

def enum(**named_values):
    return type('Enum', (), named_values)

Color = enum(RED='red', GREEN='green', BLUE='blue')
Connection_Type = enum(FOLLOWER='follower', RETWEETER='retweeter', FRIEND='friend', COCITATION='cocitation',
                       TOPIC_DISTR_SIMILARITY='topic_distr', PROFILE_PROP_SIMILARITY='profile_prop')
Author_Type = enum(BAD_ACTOR='bad_actor', GOOD_ACTOR='good_actor')
Author_Subtype = enum(PRIVATE='private',
                      COMPANY='company',
                      NEWS_FEED='news_feed',
                      SPAMMER='spammer',
                      BOT='bot',
                      CROWDTURFER='crowdturfer',
                      ACQUIRED='acquired')

CLASS_TYPE = enum(AUTHOR_TYPE='author_type', AUTHOR_SUB_TYPE='author_sub_type')

DB_Insertion_Type = enum(BAD_ACTORS_COLLECTOR='bad_actors_collector',
                         XML_IMPORTER='xml_importer',
                         MISSING_DATA_COMPLEMENTOR='missing_data_complementor',
                         BAD_ACTORS_MARKUP='bad_actors_markup',
                         MARK_MISSING_BAD_ACTOR_RETWEETERS='mark_missing_bad_actor_retweeters')

Author_Connection_Type = enum(FOLLOWER='follower', RETWEETER='retweeter', FRIEND='friend',
                              COMMON_POSTS='common_posts', COCITATION='cocitation', TOPIC='topic',
                              CITATION='citation')
Language = enum(ENGLISH='eng', GERMAN='ger')

Graph_Type = enum(CITATION='citation', COCITATION='cocitation', TOPIC='topic', FRIENDSHIP='friendship', FOLLOWER='follower')
Domains = enum(MICROBLOG='Microblog', BLOG='Blog')
Algorithms = enum(CLUSTERING='clustering', IN_DEGREE_CENTRALITY='in_degree_centrality', OUT_DEGREE_CENTRALITY='out_degree_centrality')

DistancesFromTargetedClass = enum(TRAIN_SIZE='train_size',
                                  DISTANCES_STATS='distances_statistics',
                                  MEAN='mean',
                                  MIN='min',
                                  CALCULATE_DISTANCES_FOR_UNLABELED='calculate_distances_for_unlabeled')

PerformanceMeasures = enum(AUC='AUC',
                           ACCURACY='Accuracy',
                           RECALL='Recall',
                           PRECISION='Precision',
                           CONFUSION_MATRIX='Confusion_Matrix',
                           SELECTED_FEATURES='Selected_Features',
                           CORRECTLY_CLASSIFIED="Correctly classified instances",
                           INCORRECTLY_CLASSIFIED="Incorrectly classified instances",
                           LRAP='LRAP',
                           LRL='LRL',
                           spearman_correlation='spearman_correlation',
                           spearman_coefficient_p_value='spearman_coefficient_p_value',
                           )


Classifiers = enum(RandomForest='RandomForest',
                   DecisionTree='DecisionTree',
                   AdaBoost='AdaBoost',
                   XGBoost='XGBoost')


Social_Networks = enum(TWITTER='Twitter',
                   TUMBLR='Tumblr')


AGGREGATION_FUNCTIONS = enum(AVERAGE='average',
                            MIN='min',
                            MAX='max')
