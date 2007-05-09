
import xmlrpclib
import os
import codecs
import sys 

##
## http://evanjones.ca/python-utf8.html
## http://www.jspwiki.org/Wiki.jsp?page=WikiRPCInterface2
##

print sys.argv
connect =  os.environ['TRAC_ENV_XMLRPC'] 
print  "============= connecting to remote... %s " %  connect 
server = xmlrpclib.ServerProxy(connect)

def backup(server, page ):
	''' backup file from remote wiki page to local '''
	content=server.wiki.getPage(page)
	out=file(page,'w')
	out.write( content.encode("utf-8"))
	out.close()

if len(sys.argv) < 2 :
	print "=============== backup ALL wiki pages from server to pwd: %s  " % os.environ['PWD']
	for page in server.wiki.getAllPages():
		backup( server , page )
else:
	print "============== backup specified wiki pages from server to pwd:%s  " % os.environ['PWD']
	for page in sys.argv[1:]:
		backup( server , page )

