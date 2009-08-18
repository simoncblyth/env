#!/usr/bin/env ipython
import os
def interactive_client(url=None):
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
    if url:
        r = c.get(url)
    else:
        r = None
    return c, r

if __name__=='__main__':
    import iclient 
    c,r = iclient.interactive_client("/blog/")
    os.system( "cat %s" % iclient.__file__ )

# print r 
# print r.template
# print r.context
# print r.content
