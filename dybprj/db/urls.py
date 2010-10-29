from django.conf.urls.defaults import *

urlpatterns = patterns('db.views',
    url(r'^$',                                   'db_list',    name='db-list'),
    url(r'^(?P<dbname>\w*)/$',                   'db_detail',  name='db-detail'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/$',  'db_table',   name='db-table'),
)




