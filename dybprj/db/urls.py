from django.conf.urls.defaults import *

urlpatterns = patterns('db.views',
    url(r'^$',                                                                                 'db_list',         name='db-list'),
    url(r'^(?P<dbname>\w*)/$',                                                                 'db_database',     name='db-database'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/$',                                                'db_table',        name='db-table'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/table.(?P<format>png|pdf|svg)$',                   'db_table_fig',    name='db-table-fig'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/(?P<colname>\w*)/$',                               'db_column',       name='db-column'),
    url(r'^(?P<dbname>\w*)/(?P<tabname>\w*)/(?P<colname>\w*)/column.(?P<format>png|pdf|svg)$', 'db_column_fig',   name='db-column-fig'),
)




