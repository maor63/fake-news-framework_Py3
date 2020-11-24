# Created by aviade                   
# Time: 12/04/2016 11:23

import pycurl
from io import StringIO

class WebCrawler:
    def __init__(self):
        buffer = StringIO()
        curl = pycurl.Curl()
        curl.setopt(curl.URL, 'http://pycurl.io/')
        curl.setopt(curl.WRITEDATA, buffer)
        curl.perform()
        curl.close()

        self.webpage_body = buffer.getvalue()

    def get_webpage_body(self):
        return self.webpage_body






