from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator

class Image_To_Text_Feature_Generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._post_id_text_image_dict = self._create_post_id_text_image()

    def _create_post_id_text_image(self):
        text_images = self._db.get_text_images()

        post_id_text_image_dict = {}
        for text_image in text_images:
            post_id = text_image.post_id
            post_id_text_image_dict[post_id] = text_image
        return post_id_text_image_dict