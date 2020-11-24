# Created by aviade
# Time: 29/03/2016 15:51

import logging
from configuration.config_class import getConfig
from missing_vico_posts_retriever.missing_vico_posts_retriever import MissingVicoPostsRetriever

if __name__ == '__main__':
    config_parser = getConfig()

    missing_vico_post_retriever = MissingVicoPostsRetriever()
    missing_vico_post_retriever.execute()