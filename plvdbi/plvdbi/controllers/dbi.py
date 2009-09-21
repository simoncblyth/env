
from pylons import app_globals     ## some magic here : StackedObjectProxy 

def DbiController(environ, start_response):
    return app_globals.vdbi_app(environ, start_response)
    
