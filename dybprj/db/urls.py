from django.conf.urls.defaults import *

urlpatterns = patterns('db.views',
    url(r'^$',                                        'db_list',        name='db-list'),
    url(r'^(?P<dbname>\w*)/$',                        'db_detail',      name='db-detail'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/$',       'db_table_dev',   name='db-table-dev'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/fig.(?P<format>png|pdf|svg)$', 'db_table_fig',   name='db-table-fig'),
)




