
import xmlrpclib
import sys
import os
import codecs

##
## http://evanjones.ca/python-utf8.html
## http://www.jspwiki.org/Wiki.jsp?page=WikiRPCInterface2
##

connect =  os.environ['TRAC_ENV_XMLRPC'] 
print  "============= connecting to remote... %s " %  connect 
server = xmlrpclib.ServerProxy(connect)


def restore(server, path):
	''' restore file to remote wiki '''
	if os.path.isfile(path):
		print path
		file = codecs.open( path, "r", "utf-8" )
		content = file.read()
		try:	
			server.wiki.putPage( path , content , {})
		except xmlrpclib.Fault, xf :
			#print "fault code:%s string:%s default:%s " % ( xf.faultCode , xf.faultString, xf )
			print "path:%s %s " % ( path , xf.faultString )
		except Exception, x:
			print "exception  ", x 


print sys.argv

if len(sys.argv) < 2 :
	print "=============== restore ALL backup wiki pages from pwd %s to server  " % os.environ['PWD']
	for path in os.listdir(os.environ['PWD']):
		restore( server , path )
else:
	print "============== restore specified backup wiki pages from pwd %s to server " % os.environ['PWD']
	for path in sys.argv[1:]:
		restore( server , path )
		
