
from commons.commons import *
from DB.schema_definition import Text_From_Image
import re

from preprocessing_tools.abstract_controller import AbstractController

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'


class OCR_Extractor(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._images_folder = self._config_parser.eval(self.__class__.__name__, "images_folder")

    def execute(self, window_start = None):
        start_time = time.time()
        info_msg = "execute started for " + self.__class__.__name__
        logging.info(info_msg)

        image_texts = []
        post_id_media_path_tuples = self._db.get_authors_with_media()
        self._author_screen_name_author_guid_dict = self._db.get_author_screen_name_author_guid_dictionary()

        for tuple in post_id_media_path_tuples:
            post_id = tuple[0]
            media_path = tuple[1]
            match_object = re.search('(?<=/).*', media_path)
            image_name = match_object.group(0)
            image = self._images_folder + image_name
            try:
                image_object = Image.open(image)
                text = pytesseract.image_to_string(image_object, lang='eng')

                if len(text) > 0:
                    text_from_image = Text_From_Image()
                    text_from_image.post_id = post_id

                    author_guid = self._author_screen_name_author_guid_dict[post_id]
                    text_from_image.author_guid = author_guid
                    
                    text_from_image.media_path = media_path
                    text_from_image.content = text
                    image_texts.append(text_from_image)

            except IOError as e:
                print(e.message)
                continue

        if image_texts:
            self._db.addPosts(image_texts)

        end_time = time.time()
        diff_time = end_time - start_time
        print ('execute finished in ' + str(diff_time) + ' seconds')



