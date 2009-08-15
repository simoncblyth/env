
## http://www.eflorenzano.com/blog/post/easy-multi-database-support-django/
## http://groups.google.com/group/django-users/msg/d1d7e0af565cc444?

from django.db import models
from django.conf import settings
from django.db.models import sql
from django.db.transaction import savepoint_state

try:
    import thread
except ImportError:
    import dummy_thread as thread

class MultiDBManager(models.Manager):
    """
         Select the db with 
                objects = MultiDBManager( "secondary")

    """
    use_for_related_fields = True

    def __init__(self, database, *args, **kwargs):
        self.database = database
        super(MultiDBManager, self).__init__(*args, **kwargs)

    def get_query_set(self):
        qs = super(MultiDBManager, self).get_query_set()
        qs.query.connection = self.get_db_wrapper()
        return qs

    def get_db_wrapper(self):
        database = settings.DATABASES[self.database]
        backend = __import__('django.db.backends.' + database['DATABASE_ENGINE'] + ".base", {}, {}, ['base'])
        return backend.DatabaseWrapper(database)

    def old_get_db_wrapper(self):
        database = settings.DATABASES[self.database]
        backend = __import__('django.db.backends.' + database['DATABASE_ENGINE']
            + ".base", {}, {}, ['base'])
        backup = {}
        for key, value in database.iteritems():
            backup[key] = getattr(settings, key)
            setattr(settings, key, value)
        wrapper = backend.DatabaseWrapper()
        wrapper._cursor(settings)
        for key, value in backup.iteritems():
            setattr(settings, key, value)
        return wrapper

    def _insert(self, values, return_id=False, raw_values=False):
        query = sql.InsertQuery(self.model, self.get_db_wrapper())
        query.insert_values(values, raw_values)
        ret = query.execute_sql(return_id)
        query.connection._commit()
        thread_ident = thread.get_ident()
        if thread_ident in savepoint_state:
            del savepoint_state[thread_ident]
        return ret






