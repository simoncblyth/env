from django.db.models.signals import post_syncdb
from dybprj.db.bootstrap import bootstrap

def bootstrap_callback(sender, **kwa):
    app = kwa.get('app', None)
    if app and app.__name__ == 'dybprj.db.models':
        print "bootstrapping %s " % app.__name__
        bootstrap()
    else:
        print "skip %s " % app.__name__
    pass
post_syncdb.connect(bootstrap_callback)






