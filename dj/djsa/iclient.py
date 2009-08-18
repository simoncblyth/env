#!/usr/bin/env ipython
import os
def interactive_client():
    """
          http://simonwillison.net/2008/May/22/debugging/

         NB due to fixing DJANGO_SETTINGS_MODULE this must 
         be run from the project folder under test
    """
    os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
    from django.test.utils import setup_test_environment
    setup_test_environment()
    from django.test.client import Client
    c = Client()
    return c

if __name__=='__main__':
    import iclient 
    c = iclient.interactive_client()
    os.system( "cat %s" % iclient.__file__ )

# r = c.get("/blog/")
# print r 
# print r.template
# print r.context
# print r.content
