
import settings
from djsa.blog.models import Post
m = Post._default_manager
qs = m.get_query_set()
print qs 


