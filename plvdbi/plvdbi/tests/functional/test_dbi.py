from plvdbi.tests import *

class TestDbiController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='dbi', action='index'))
        # Test response...
