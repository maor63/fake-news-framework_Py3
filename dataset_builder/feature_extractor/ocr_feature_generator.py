# Created by Aviad Elyashar at 19/09/2017
from dataset_builder.feature_extractor.image_to_text_feature_generator import Image_To_Text_Feature_Generator

class OCR_Feature_Generator(Image_To_Text_Feature_Generator):
    def __init__(self, db, **kwargs):
        Image_To_Text_Feature_Generator.__init__(self, db, **kwargs)

    def cleanUp(self):
        pass

    def is_post_contain_image(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            media_path = author.media_path
            if media_path is None:
                return 0
            return 1
        else:
            raise Exception('Author object was not passed as parameter')

    def is_image_contain_text(self, **kwargs):
        if 'author' in list(kwargs.keys()):
            author = kwargs['author']
            post_id = author.name
            if post_id in self._post_id_text_image_dict:
                return 1
            return 0
        else:
            raise Exception('Author object was not passed as parameter')
