
from django.core.management.base import NoArgsCommand
from django.core.management import call_command
from django.conf import settings

from multidb.util.manager import MultiDBManager


class Command(NoArgsCommand):
    help = "Sync multiple databases."

    def handle_noargs(self, **options):
        verbosity = int(options.get('verbosity', 1))
        for name, database in settings.DATABASES.iteritems():
            if verbosity > 0:
                print "Running syncdb for %s " % (name,)
            manager = MultiDBManager(name)
            import django.db as db  
            setattr( db , 'connection' ,  manager.get_db_wrapper() )
            call_command('syncdb', verbosity=verbosity )



