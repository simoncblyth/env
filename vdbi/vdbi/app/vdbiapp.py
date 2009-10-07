#from __future__ import with_statement  ## not available in 2.4
#from vdbi.dbg import debug_here

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
    logc = { 'rum.basefactory':logging.INFO , 'vdbi.rum.query':logging.DEBUG , 'vdbi.rumalchemy.repository':logging.DEBUG }
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


def create_app(url=None,  dbg=True):
    from env.base.private import Private
    p = Private()
    if not(url):
       url = p('DATABASE_URL')          
    logdir = p('VDBI_LOGDIR') 

    print "create_app logdir:%s " % logdir 
    setup_logging(logdir)
    
    import rum.util
    rum.util.generate_label = lambda x:x   ## stomp on the decamelization 
    
    from pkg_resources import resource_filename
    import os
    from rum import RumApp
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


    ## setup the JSON specialized controller
    from vdbi.rum.controller import register
    register()


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
    for cls in app.resources.keys():
        for f in app.fields_for_resource( cls ):
            f.searchable = True
            #f.read_only = True
            f.auto = False       ## succeeds to get ROW_COUNTER to appear on payload tables and SEQNO to appear on Vld tables 
            f.label = f.name    
            #print f


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

 
