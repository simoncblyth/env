
from django.conf import settings
from django.db.models.signals import post_syncdb
from models import * 

def bootstrap():
    print "bootstraping from settings.DATABASES "
    for k in filter(lambda _:_ != "default",  settings.DATABASES.keys()):
        print k
        Database.objects.get_or_create( name=k )

def bootstrap_callback(sender, **kwa):
    app = kwa.get('app', None)
    if app and app.__name__ == 'dbi.models':
        bootstrap()

post_syncdb.connect(bootstrap_callback)




