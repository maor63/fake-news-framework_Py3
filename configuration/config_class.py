# Created by aviade      
# Time: 09/05/2016 17:31

__author__ = 'dana'
import configparser
import sys
import logging
from commons.commons import * #common and builtin functions can be used in config values
__config =None
__config_file = None

def getConfig():
    """
    @description: config file given as the first commnand line param.
    :return: config instance
    """
    global __config, __config_file
    if __config and __config_file==sys.argv[1]:
        return __config
    else:
        try:
            __config_file = sys.argv[1]
            logging.info("config file %s"%__config_file)
            __config = configparser.ConfigParser()
            __config.read(__config_file, encoding="utf8")
            __config.eval = lambda sec,key: eval(__config.get(sec,key))
            __config.getfilename = lambda: __config_file
            return __config
        except:
            logging.exception( "Usage: %s <path to config .ini file>" , (sys.argv[0]))
            exit(-1)



########################################################################
import unittest
class TestConfig(unittest.TestCase):
    def setUp(self):
        import sys;sys.argv = [sys.argv[0], 'config_test_offline.ini']
        self._config= getConfig()

    def tearDown(self):
        pass

    def testDefaultValue(self):
        self.assertEqual(self._config.get("DB","start_date"), "date('2015-04-01 00:00:00')")

    def testAccessDefault(self):
        self.assertEqual(self._config.get("DEFAULT","start_date"), "date('2015-04-01 00:00:00')")

    def testEval(self):
        import datetime
        self.assertEqual(self._config.eval("DEFAULT","start_date"), datetime.datetime.strptime('2015-04-01 00:00:00',"%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    unittest.main()

