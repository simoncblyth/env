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
    for cls,opt in app.resources.items():
        for f in app.fields_for_resource( cls ):
            f.searchable = True
            f.read_only = True
            f.auto = False       ## succeeds to get ROW_COUNTER to appear on payload tables and SEQNO to appear on Vld tables 
            print f


if __name__=='__main__':
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    opts, args = parser.parse_args(sys.argv)

    app = load_app(opts.url,  debug=True )
    field_fix( app )
    app.finalize()
    rf = app.repositoryfactory  
    for cls in rf.resources.keys():print repr(cls._table.c.SEQNO)

    server = loadserver('egg:Paste#http')
    try:
        server(app)
    except (KeyboardInterrupt, SystemExit):
        print "Bye!"



