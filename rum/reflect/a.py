import sys
import logging
from optparse import OptionParser

from sqlalchemy import create_engine, MetaData
from paste.deploy import loadserver

 
from rum import RumApp


from tw.rum import RumDataGrid
RumDataGrid.actions = ['show']


#
# A parser for command line options
#

from env.base.private import Private
p = Private()
dburl = p('DATABASE_URL')  ## sqlite:///movie.db

parser = OptionParser()
parser.add_option('', '--dburl',
                  dest='url',
                  help='SQLAlchemy database uri (eg: postgres:///somedatabase)',
                  default=dburl )
parser.add_option('-d', '--debug',
                  dest='debug',
                  help='Turn on debug mode',
                  default=False,
                  action='store_true')

#
# Makes the app
#
def load_app(url,  debug=False):
    app = RumApp({
        'debug': debug,
        'default_page_size':30,
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
    #app.finalize()
    return app

#
# Main calling point
#


def field_fix( app ):
    """
        attempt to make the fields readonly and all displayed ...
    """
    for cls in app.resources.keys():
        for f in app.fields_for_resource( cls ):
            f.searchable = True
            f.read_only = True
            f.auto = False       ## succeeds to get ROW_COUNTER to appear on payload tables and SEQNO to appear on Vld tables 
            print f



class Mapr(dict):
    """
        For checking on the mappers that the app is using 
    """
    def __init__(self, app):
        from rumalchemy.util import get_mapper
        for cls in app.resources.keys():
            if hasattr(cls._table.c, 'SEQNO' ):
                print repr(cls._table.c.SEQNO)
            mapr = get_mapper(cls)
            self[cls.__name__] = mapr
            print mapr.c.keys()

class Qry(dict):
    """
        For doing test queries using the session etc... hooked up into the app 

      In [2]: q['Simpmtspec']
      Out[2]: <sqlalchemy.orm.query.Query object at 0x1e31f50>
      In [3]: q['Simpmtspec'].first()
      In [4]: q['Simpmtspec'].count()
      Out[4]: 3169L

    """
    def __init__(self, app ): 
        sf = app.repositoryfactory.session_factory
        for cls in app.resources.keys():
            self[cls.__name__] = sf.query(cls)
            print sf.query(cls).first()






class Repo(dict):
    """
           r = Repo(app)
           print r['Simpmtspec'].get( (1,1 ) )   ## CPK get 

         

    """
    def __init__(self, app ):
        for cls in app.resources.keys():
            repo = app.repositoryfactory( cls )
            self[cls.__name__] = repo
            
            if cls.__name__ == 'Simpmtspec':
            	qf = repo.queryfactory.get( cls )
            	from rum.query import Query, eq 
                
                # should not need to descend to the SA specialization 
                # from rumalchemy.query import SAQuery
            	# saq = SAQuery( eq('ROW', 10 ) , resource=cls ) 
            	# print list( saq.filter( qs['Simpmtspec'] ) )

                q = Query( eq('ROW', 10 ) )
                qq = repo.make_query( request_args=q.as_dict() )
                ss = repo.select(qq)
                print list(ss)

    def query( self , repo , q ): 
        return repo.select( repo.queryfactory( repo.resource , request_args = q.as_dict() ) )





if __name__=='__main__':
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    opts, args = parser.parse_args(sys.argv)

    app = load_app(opts.url,  debug=True )
    field_fix( app )
    app.finalize()

    m = Mapr( app )
    q = Qry(  app )
    r = Repo(  app )

    server = loadserver('egg:Paste#http')
    try:
        server(app)
    except (KeyboardInterrupt, SystemExit):
        print "Bye!"


