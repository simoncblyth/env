"""
   Standalone-ish running requires : 
        1) envvar DJANGO_SETTINGS_MODULE eg dybprj.settings 
        2) dybprj on sys.path (ie with dybprj-ln) 
        3) "py absolute" module refs in settings.py , eg "dybprj.db" rather than "db" 

"""
from django.conf import settings
from dybprj.db.models import * 
from django.db import connections

from django.contrib.sites.models import Site


# this is very generic ... it does not belong here 
def set_site_domain():
    s = Site.objects.get_current()
    s.name = settings.SITE_DOMAIN
    s.domain = settings.SITE_DOMAIN
    s.save()
    print "set_site_domain to %s " % s


def populate():
    print "bootstraping from connections.databases "
    for k in filter( lambda k:not k == "default", connections.databases.keys()):
        cfg = connections.databases[k]  
        db, created = Database.objects.get_or_create( name=k )
        print k, cfg
        print db 
        cur = connections[k].cursor()
        cur.execute("show tables");     
        for tn in cur.fetchall(): 
            tab = Table.objects.get_or_create( name=tn[0] , db=db )
            print tab

def bootstrap():
    set_site_domain()
    populate()


if __name__ == '__main__':
    import os
    assert os.environ.get("DJANGO_SETTINGS_MODULE", None ), __doc__  
    bootstrap()
