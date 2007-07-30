
import xmlrpclib
import sys
import os
import codecs
import re

##
## http://evanjones.ca/python-utf8.html
## http://www.jspwiki.org/Wiki.jsp?page=WikiRPCInterface2
##


argv=sys.argv[1:]  
envname=argv.pop(0)
print argv

## replace env with the trac environment name from the first argument
patn=re.compile('(.*\/)env(\/.*)') 
tmpl=os.environ['TRAC_ENV_XMLRPC']
mtch=patn.match(tmpl) 
connect='%s%s%s' % ( mtch.group(1) , envname , mtch.group(2) )  
print  "============= [%s] connecting to remote... %s " % ( envname , connect )
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


print argv

if len(argv) == 0 :
	print "=============== restore ALL backup wiki pages from pwd %s to server  " % os.environ['PWD']
	for path in os.listdir(os.environ['PWD']):
		restore( server , path )
else:
	print "============== restore specified backup wiki pages from pwd %s to server " % os.environ['PWD']
	for path in argv:
		restore( server , path )
		
