from django.conf.urls.defaults import *

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

from django.contrib import databrowse

# http://code.djangoproject.com/ticket/10061
# temporary fix to make the logout button work 
admin.site.root_path = '/dybsite/admin/'



urlpatterns = patterns('',
    # Example:
    # (r'^dybsite/', include('dybsite.foo.urls')),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
     (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
     (r'^admin/', include(admin.site.urls)),

     (r'^databrowse/(.*)', databrowse.site.root),


)
