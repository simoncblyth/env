import sys
import logging
from optparse import OptionParser

from sqlalchemy import create_engine
from paste.deploy import loadserver

from rum import RumApp

from model import Model, Person, Genre, Actor, Director, Movie, Rental

#
# A parser for command line options
#
parser = OptionParser()
parser.add_option('', '--dburl',
                  dest='url',
                  help='SQLAlchemy database uri (eg: postgres:///somedatabase)',
                  default='sqlite:///movie.db')
parser.add_option('-d', '--debug',
                  dest='debug',
                  help='Turn on debug mode',
                  default=False,
                  action='store_true')

#
# Makes the app
#
def load_app(url, debug=False):
    models = [Person, Genre, Actor, Director, Movie, Rental]
    return RumApp({
        'debug': debug,
        'rum.repositoryfactory': {
            'use': 'sqlalchemy',
            'models': models,
            'sqlalchemy.url': url,
            'session.transactional': True,
        },
        'rum.viewfactory': {
            'use': 'tutetoscawidgets',
        }
    })

#
# Main calling point
#
def main(argv=None):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    opts, args = parser.parse_args(argv)
    Model.metadata.create_all(bind=create_engine(opts.url))
    app = load_app(opts.url, opts.debug)
    server = loadserver('egg:Paste#http')
    try:
        server(app)
    except (KeyboardInterrupt, SystemExit):
        print "Bye!"
    return app


if __name__ == '__main__':
    #sys.exit(main(sys.argv))
    app = main(sys.argv)


