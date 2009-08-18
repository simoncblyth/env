from django.test import TestCase
from django.test.client import Client

class TestStandardUrlConf(TestCase):
    def _test_index_missing(self):
        c = Client()
        resp = c.get('')
        assert resp.status_code == 404, "Got status %s - expecting 404" % resp.status_code


    def test_index(self):
        c = Client()
        resp = c.get('/blog/')
        
        assert "Select post to change" in resp.content, resp.content




