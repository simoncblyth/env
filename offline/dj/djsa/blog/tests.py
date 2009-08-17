"""
This file demonstrates two different styles of tests (one doctest and one
unittest). These will both pass when you run "manage.py test".

Replace these with more appropriate tests for your application.
"""

from django.test import TestCase

class SimpleTest(TestCase):
    def test_basic_addition(self):
        """
        Tests that 1 + 1 always equals 2.
        """
        self.failUnlessEqual(1 + 1, 2)

__test__ = {"doctest": """
Another way to test that 1 + 1 is equal to 2.

>>> 1 + 1 == 2
True
"""}

def test_trivial():

    from django_sqlalchemy.backend import metadata
    from djsa.blog.models import Trivial 
    metadata.create_all()
    p = Trivial(title="the title", body="the body")
    p.save()


def test_post():
    from djsa.blog.models import Post
    p = Post(title="title", body="body")
    p.save()

if __name__=='__main__':
    #test_trivial()
    test_post()
