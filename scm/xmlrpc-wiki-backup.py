
import xmlrpclib
import os
import codecs
import sys 
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


def backup(server, page ):
    ''' backup file from remote wiki page to local '''
    content=server.wiki.getPage(page)
    out=file(page,'w')
    out.write( content.encode("utf-8"))
    out.close()
 
        
if len(argv) == 0 :
	print "=============== backup ALL wiki pages from server to pwd: %s  " % os.environ['PWD']
	for page in server.wiki.getAllPages():
		backup( server , page )
else:
	print "============== backup specified wiki pages from server to pwd:%s  " % os.environ['PWD']
	for page in argv:
		backup( server , page )

