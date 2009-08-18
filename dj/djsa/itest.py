#!/usr/bin/env ipython

"""
   http://simonwillison.net/2008/May/22/debugging/
"""
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'

from django.test.utils import setup_test_environment
setup_test_environment()



from django.test.client import Client
c = Client()


#r = c.get("/blog/")

#print r 
#print r.template
#print r.context
