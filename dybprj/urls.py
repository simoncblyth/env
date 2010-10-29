from django.conf.urls.defaults import *

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',

    # Uncomment the admin/doc line below to enable admin documentation:
     (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
     (r'^admin/', include(admin.site.urls)),

    # enabling comment pages ... preview/moderation etc..
     (r'^comments/', include('django.contrib.comments.urls')),

    # enabling comment pages ... preview/moderation etc..
     (r'^db/', include('db.urls')),

    # default login form 
     (r'^accounts/login/$', 'django.contrib.auth.views.login'),

)

#from django.conf import settings
#if settings.DEBUG:
#    urlpatterns += url(r'^static/(?P<path>.*$)', 'django.views.static.serve', {'document_root': settings.MEDIA_ROOT } , name="static"  ),

