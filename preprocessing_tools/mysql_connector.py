# Created by aviade      
# Time: 29/05/2016 10:55

from config_class import getConfig
import MySQLdb
from .abstract_controller import AbstractController

class MysqlConnector(AbstractController):
    '''
    Abstract class for classes interfacing with mysql
    '''
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self._configInst = getConfig()
        self.host = self._configInst.get(self.__class__.__name__, "host")
        self.user = self._configInst.get(self.__class__.__name__, "user")
        self.pwd = self._configInst.get(self.__class__.__name__, "pwd")
        self.schema = self._configInst.get(self.__class__.__name__, "schema")
        self._mysql_conn = None

    def setUp(self):
        AbstractController.setUp(self)

    def tearDown(self):
        AbstractController.tearDown(self)
        if  self._mysql_conn:
            self.mysql_close()

    def execute(self, window_start):
        AbstractController.execute(self, window_start)

    def mysql_open(self):
        if not self._mysql_conn:
            self._mysql_conn = MySQLdb.connect(host=self.host,
                                               user=self.user,
                                               passwd=self.pwd,
                                               db=self.schema,
                                               charset='utf8',
                                               use_unicode=True)
        self.c = self._mysql_conn.cursor()

    def mysql_close(self):
        if self._mysql_conn is not None:
            self._mysql_conn.commit()
            self._mysql_conn.close()
            self.c = None
            self._mysql_conn = None

    def execute_mysql_query(self, query, params = None):
        if params is None:
            self.c.execute(query)
        else:
            self.c.execute(query, params)
        data = self.c.fetchall()
        self.c.nextset()
        return data

    def get_mysql_literal(self, text):
        return self._mysql_conn.literal(text)