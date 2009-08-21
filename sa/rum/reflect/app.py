import sys
import logging
from optparse import OptionParser

from sqlalchemy import create_engine, MetaData
from paste.deploy import loadserver
from sqlalchemy.orm import scoped_session, sessionmaker
Session = scoped_session(sessionmaker(autocommit=False, autoflush=True))


 
from rum import RumApp

#from model import Model, Person, Genre, Actor, Director, Movie, Rental

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
def load_app(url, reflect, debug=False):
    app = RumApp({
        'debug': debug,
        'rum.repositoryfactory': {
            'use': 'sqlalchemy',
            'reflect':reflect  ,
          #  'models': models,
            'sqlalchemy.url': url,
            'session_factory':'app:Session',  
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



def dbi_fkc( metadata ):
    """
       Append FK constraints to the SA tables as MySQL dont know em 

    """
    from sqlalchemy import ForeignKeyConstraint
    pay_tables = [n[0:-3] for n,t in metadata.tables.items() if n.endswith('Vld')]
    vld_tables = ["%sVld" % n for n in pay_tables]    
    for p,v in zip(pay_tables,vld_tables): 
        pay = metadata.tables.get(p, None )
        vld = metadata.tables.get(v, None )
        if not(pay) or not(vld):
            print "skipping tables %s " % n
            continue
        pay.append_constraint( ForeignKeyConstraint( ['SEQNO'] , ['%s.SEQNO' % v ] ) )
    return pay_tables + vld_tables 


def readonly( app ):
    """
        attempt to make the fields readonly and all displayed ...
    """
    for cls,opt in app.resources.items():
        for f in app.fields_for_resource( cls ):
            f.searchable = True
            f.read_only = True
            print f




if __name__=='__main__':

    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    opts, args = parser.parse_args(sys.argv)

    ## 
    ## try setting up the Session first in the hope that Soup will reuse the fixed up tables
    ## see rumalchemy/test_repository.py 
    ##
    engine = create_engine( opts.url )
    Session.configure(bind=engine)

    metadata = MetaData( engine )
    metadata.reflect()   

    tables = dbi_fkc( metadata )
    ## the FK is there 
    print metadata.tables['SimPmtSpec']
    print metadata.tables['SimPmtSpecVld']




    #metadata.create_all()
    app = load_app(opts.url, tables , opts.debug )
    #readonly( app )
    app.finalize()

    rf = app.repositoryfactory
    ## but the FK is not here ...  so the reflection is creating new tables 
    
    for cls in rf.resources.keys():print repr(cls._table.c.SEQNO)

    assert rf.session_factory == Session  


    server = loadserver('egg:Paste#http')
    try:
        server(app)
    except (KeyboardInterrupt, SystemExit):
        print "Bye!"



