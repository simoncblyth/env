


def load_app(url=None,  dbg=True):

    if not(url):
        from env.base.private import Private
        p = Private()
        url = p('DATABASE_URL')          
 
    from rum import RumApp
    app = RumApp({
        'debug': dbg,
        'default_page_size':30,
        'rum.policy':{
            'use': 'vdbipolicy' ,
        },
        'rum.repositoryfactory': {
            'use': 'vdbisqlalchemy',
            'reflect':'dbi'  ,
            'sqlalchemy.url': url,
            'session.transactional': True,
        },
        'rum.viewfactory': {
            'use': 'toscawidgets',
        }
    }, finalize=False )

    field_fix( app )

    from tw.rum import RumDataGrid
    RumDataGrid.actions = ['show']

    if dbg:
        from debug import Repo, Qry , Mapr, Dump
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
            f.read_only = True
            f.auto = False       ## succeeds to get ROW_COUNTER to appear on payload tables and SEQNO to appear on Vld tables 
            print f


def serve_app(**kwargs):
    from vdbi.app import load_app
    app = load_app( **kwargs )
    from paste.deploy import loadserver
    server = loadserver('egg:Paste#http')
    try:
        server(app)
    except (KeyboardInterrupt, SystemExit):
        print "Bye! returning app"
    return app



if __name__=='__main__':
    #app = load_app(dbg=True)
    app = serve_app(dbg=True)

    ## the below is almost equivalent to :   run -d 
    ## import pdb
    ## pdb.run("from vdbi import serve_app ; app = serve_app() ")



 
