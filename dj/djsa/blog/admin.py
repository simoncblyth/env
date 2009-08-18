from django.contrib import admin
from djsa.blog.models import Post

class PostAdmin(admin.ModelAdmin):
    list_display = ('title','body','date_submitted',) 

admin.site.register(Post, PostAdmin)



