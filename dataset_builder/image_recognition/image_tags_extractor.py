#
# Created by Aviad on 03-Jun-16 11:41 AM.
#
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from preprocessing_tools.abstract_controller import AbstractController
from DB.schema_definition import Image_Tags


class Image_Tags_Extractor(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._auth_key = self._config_parser.eval(self.__class__.__name__, "auth_key")
        self._model_id = self._config_parser.eval(self.__class__.__name__, "model_id")
        self._max_objects_without_saving = self._config_parser.eval(self.__class__.__name__, "max_objects_without_saving")
        self._max_request_to_service = self._dataset_folder = self._config_parser.eval(self.__class__.__name__, "max_request_to_service")
        self._dataset_folder = self._config_parser.eval(self.__class__.__name__, "dataset_folder")
        self.min_value_proba = self._config_parser.eval(self.__class__.__name__, "min_value_proba")

    def _write_tags_to_DB(self, to_write):
        self._db.addPosts(to_write)
        ls = []
        return ls

    def _get_tags(self, post_images):
        app = ClarifaiApp(api_key=self._auth_key)
        model = app.models.get(self._model_id)
        try:
            tags = model.predict(post_images)["outputs"]
            authors_tags = []
            for tag in tags:
                author_tags = tag["data"]["concepts"]
                author_tags = [key["name"] for key in author_tags if key["value"] > self.min_value_proba]
                authors_tags.append(author_tags)
            return authors_tags

        except:
            print('possible problem, clarifai queries limit exceeded')

    def setUp(self):
        pass

    def execute(self, window_start):

        authors = self._db.get_authors_with_media()
        post_images_to_send = []
        image_tags = []
        requests_counter = 0
        flush_counter = 0
        for author in authors:

             image_tag = Image_Tags()
             image_tag.post_id = author[0]
             image_tag.media_path = author[1]
             image_tags.append(image_tag)

             opened_image = ClImage(file_obj=open(self._dataset_folder + author[1], 'rb'))


             if requests_counter == self._max_request_to_service:
                 requests_counter = 0
                 tags = self._get_tags(post_images_to_send) #might return an error if limit is exceeded
                 post_images_to_send = []

                 #adding the tags to the objects list
                 for j in range(self._max_request_to_service):
                        image_tags[flush_counter].tags = ", ".join(tags[j])
                        flush_counter+=1

             if flush_counter >= self._max_objects_without_saving: #writing the info to the DB
                 image_tags = self._write_tags_to_DB(image_tags)
                 flush_counter = 0
             post_images_to_send.append(opened_image)
             requests_counter += 1
        #flushing the leftovers if left
        print(str(len(post_images_to_send)) + ' left')
        if len(post_images_to_send)>0:
            tags = self._get_tags(post_images_to_send) #might return an error if limit is exceeded
            post_images_to_send = []
            for j in range(len(tags)):
                image_tags[flush_counter].tags = ", ".join(tags[j])
                flush_counter += 1
            image_tags = self._write_tags_to_DB(image_tags)
        flush_counter = 0
        requests_counter = 0



