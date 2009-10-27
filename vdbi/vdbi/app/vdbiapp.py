#from __future__ import with_statement  ## not available in 2.4
#from vdbi.dbg import debug_here

import logging
log = logging.getLogger(__name__)

import time
tfmt = lambda t:time.strftime("%Y%m%d",time.localtime(t))

def stat_( path ):
    import os, stat
    if not os.path.exists(path):return {}
    s = os.stat(path)
    return {
      'mtime': tfmt(s[stat.ST_MTIME]),
      'atime': tfmt(s[stat.ST_ATIME]),
      'ctime': tfmt(s[stat.ST_CTIME]),
    }


def handle_log( name , logdir ):
    import logging
    import logging.handlers
    import os
    log = logging.getLogger( name )
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    handler = logging.handlers.RotatingFileHandler( os.path.join( logdir, '%s.log' % name ) , maxBytes=1000000 , backupCount=5)
    log.addHandler(handler)
    return log 

def setup_logging(logdir):
    import logging
    logc = { 'rum.basefactory':logging.INFO , 
             'vdbi.rum.controller':logging.DEBUG,
             'vdbi.rum.query':logging.DEBUG ,
             'vdbi.rumalchemy.repository':logging.DEBUG }
    #logc.update(**kw)
    #logging.basicConfig()
    for name,levl  in logc.items():
        handle_log( name , logdir=logdir ).setLevel( levl )
    

from rum.controller import Controller
class LoginController(Controller):
    def index(self):
        self.template = 'dbilogin'
        print "login controller "
        return {'msg':"from the login controller"}

class RootController(Controller):
    def index(self):
        self.template = 'home'
        return {}



from rum.fields import Relation
from rum import RumApp
def related_resources(self, resource):
    """
        Not done in a RumApp subclass, as (from rum import app) is used everywhere 
        and the app a threadsafe singleton entity, so probably tricky to diddle with it
    """
    return [f.other for f in self.fields_for_resource( resource ) if issubclass(f.__class__, Relation) and f.other != resource]

def paired_resource(self, resource, dbi_or_vld=True ):
    related = self.related_resources( resource )
    if dbi_or_vld:
        for r in related:  
            if r.__name__[-3:] not in ('Dbi','Vld',):
                related.remove(r)  
    if len(related) == 1:
        return related[0]
    elif len(related) > 1 :
        return related[0]
    else:
        return None
 
RumApp.related_resources = related_resources
RumApp.paired_resource = paired_resource


def create_app(url=None,  dbg=True):
    from private import Private
    p = Private()
    if not(url):
       url = p('DATABASE_URL')          
    logdir = p('VDBI_LOGDIR') 

    print "create_app private config from : %s " % p.path 
    setup_logging(logdir)
    
    import rum.util
    rum.util.generate_label = lambda x:x   ## stomp on the decamelization 
    
    from pkg_resources import resource_filename, get_distribution
    import os
    
    app = RumApp({
        'debug': dbg,
        'full_stack':True,
        'default_page_size':30,
        'widgets': {
            'rum_css':'vdbi.rum.widgets:rum_css',
        },
         'templating': {
                'search_path': [os.path.abspath(resource_filename('vdbi.rum','templates'))] , 
         },
        'rum.policy':{
            'use': 'vdbipolicy' ,
        },
        'rum.repositoryfactory': {
            'use': 'vdbisqlalchemy',
            'reflect':'dbi'  ,
            'sqlalchemy.url': url,
            'session.autocommit': True,
        },
        'rum.viewfactory': {
            'use': 'vdbitoscawidgets',
        }
    }, finalize=False, root_controller=RootController )

    field_fix( app )

    ## record the versions of the principal packages for quoting in the footer    
    app.pkgs = []
    vdbi_packages = p('VDBI_PACKAGES')
    pkgs = "%s" % vdbi_packages.replace('"','')
    for pkg in pkgs.split(','):
        print pkg
        try:
            dist = get_distribution(pkg)
            app.pkgs.append(dist)
        except:
            print "failed to get_distribution for %s" % pkg

    ## report the date on the statics dir
    plvdbi_statics_dir = p('PLVDBI_STATICS_DIR')
    dtfmt = stat_(plvdbi_statics_dir)
    app.pkgs.append("statics %s" % dtfmt['mtime'] )

    print "\n".join([repr(p) for p in app.pkgs])
    

    ## setup the JSON specialized controller
    from vdbi.rum.controller import register_crud_controller
    register_crud_controller()

    app.router.connect("login/:action", controller=LoginController )


    if dbg:
        from vdbi.app.debug import Repo, Qry , Mapr, Dump
        app.__class__._repo = lambda self:Repo(self)
        app.__class__._qry  = lambda self:Qry(self)
        app.__class__._mapr = lambda self:Mapr(self)
        app.__class__._dump = lambda self:Dump(self)

    app.finalize()
    return app

def field_fix( app ):
    
    from rum.fields import Relation
    for cls in app.resources.keys():
        log.debug("field_fix for cls : %s " % cls)
        for f in app.fields_for_resource( cls ):
            f.searchable = not(issubclass(f.__class__, Relation)) 
            f.read_only = True
            f.auto = False           ## succeeds to get ROW_COUNTER to appear on payload tables and SEQNO to appear on Vld tables 
            f.label = f.name   
            f.plotable = not(issubclass(f.__class__, Relation)) 
            #print f, "plotable:", f.plotable


def serve_app(**kwa):
    from vdbi.app import create_app
    port = kwa.pop('port', 8080 )
    app = create_app( **kwa )
    from paste.deploy import loadserver
    from pkg_resources import resource_filename
    ini = resource_filename('vdbi.app', 'server.ini')
    print "serve_app with ini %s " % ini
    #server = loadserver('egg:Paste#http' )
    server = loadserver('config:%s' % ini  )
    try:
        server(app)
    except (KeyboardInterrupt, SystemExit):
        print "Bye! returning app"
    return app



if __name__=='__main__':


    #vdbi_app = create_app(dbg=True)
    vdbi_app = serve_app(dbg=True)

    ## the below is almost equivalent to :   run -d 
    ## import pdb
    ## pdb.run("from vdbi import serve_app ; app = serve_app() ")


    ## http://docs.python-rum.org/tip/developer/modules/index.html
    ## using "with" (from future) enables rum.app to stay non-None outside of WSGI request context
    ##
    ##  
    ##with vdbi_app.mounted_at("/"):
    ##    print vdbi_app.viewfactory   ## <vdbi.tw.rum.viewfactory.DbiWidgetFactory object at 0x21b4090>

 
