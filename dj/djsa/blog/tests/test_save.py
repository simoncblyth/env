
def test_trivial_sa():
    from django.conf import settings
    if not(settings.DATABASE_ENGINE.startswith('django_sqlalchemy')):return 
    from django_sqlalchemy.backend import metadata
    from djsa.blog.models import Trivial 
    metadata.create_all()
    p = Trivial(title="the title", body="the body")
    p.save()

def test_trivial():
    from djsa.blog.models import Trivial 
    p = Trivial(title="the title", body="the body")
    p.save()

def test_post():
    from djsa.blog.models import Post
    p = Post(title="title", body="body")
    p.save()

if __name__=='__main__':
    #test_trivial()
    test_post()
