if __name__=='__main__':
    import os
    os.environ['DJANGO_SETTINGS_MODULE'] = 'djsa.settings'
 

from djsa.blog.models import Post

def test_qs():
    m = Post._default_manager
    qs = m.get_query_set()
    print qs 

if __name__=='__main__':
    test_qs()
