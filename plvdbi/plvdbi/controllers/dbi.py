
from pylons import config
g = config['pylons.app_globals']

def DbiController(environ, start_response):
    return g.vdbi_app(environ, start_response)
    
