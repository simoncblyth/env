
from django.core.management.base import NoArgsCommand
from django.core.management import call_command
from django.conf import settings

class Command(NoArgsCommand):
    help = "Sync multiple databases."

    def handle_noargs(self, **options):
        for name, database in settings.DATABASES.iteritems():
            print "Running syncdb for %s " % (name,)
            for key, value in database.iteritems():
                print "setting %s %s " % ( key, value )
                setattr(settings, key, value)
            #from django.conf import settings
            from django.db import connection
            print "connection settings: %s connection:%s " % ( connection.settings_dict, connection )
            call_command('syncdb')



