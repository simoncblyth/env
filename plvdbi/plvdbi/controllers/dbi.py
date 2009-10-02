
from pylons import app_globals     ## some magic here : StackedObjectProxy 

def DbiController(environ, start_response):
    keys = ('SCRIPT_NAME', 'PATH_INFO', 'REQUEST_URI',)
    print "DbiController %s " % " ".join( ["%s = %s " % ( k , environ.get(k,None) ) for k in keys ])
    return app_globals.vdbi_app(environ, start_response)
    
