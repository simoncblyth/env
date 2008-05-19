

modwsgi-use-env(){

   elocal-
   trac-
   python-

}


modwsgi-use-app(){
   
   name=${1:-dummy}
#
#  TRAC_ENV points to the trac site instance 
#
cat << EOA
import os
os.environ['TRAC_ENV'] = '$SCM_FOLD/tracs/$name'
os.environ['PYTHON_EGG_CACHE'] = '$SCM_FOLD/tracs/$name/eggs'
import trac.web.main
#application = trac.web.main.dispatch_request

## chevron printing, the first expr  must evaluate to a file object ... like a log
## to which the subsequent expr are sent
## print >> environ['wsgi.errors'], environ 

def application(environ, start_response): 
	for k, v in sorted(environ.iteritems()):
		print >> environ['wsgi.errors'], "   %-25s : %s " %  ( k,  v )
	return trac.web.main.dispatch_request(environ, start_response) 


EOA
}


