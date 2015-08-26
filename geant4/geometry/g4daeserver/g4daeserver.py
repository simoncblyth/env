#!/usr/bin/env python
"""

Usage::

   g4daeserver.py --help
   g4daeserver.py             # default of scgi for apache on G 
   g4daeserver.py -w fcgi     # fastcgi for nginx on N 

   g4daeserver.py --daepath $LOCAL_BASE/env/graphics/collada/3199.dae  
      # starting from a small DAE file is convenient for parsing speed during development
      # NB volume indices will then be relative to that sub-root  

With webpy SCGI deployment with apache

* http://localhost/g4dae/hello/hello.html?name=simon
* http://localhost/g4dae/tree/0___0.html
* http://localhost/g4dae/tree/3199___0.html
* http://localhost/g4dae/tree/0___0.dae

Basis Geometry Files
----------------------

::

    [blyth@belle7 g4daeserver]$ pwd
    /data1/env/local/env/geant4/geometry/g4daeserver
    [blyth@belle7 g4daeserver]$ cp ../gdml/VDGX_20131121-2043/g4_00.dae VDGX_20131121-2043_g4_00.dae
    [blyth@belle7 g4daeserver]$ cp ../gdml/DVGX_20131121-2053/g4_00.dae DVGX_20131121-2053_g4_00.dae 


PUT issue
-----------

Trying to recive PUTs from shift gives::

    2013-12-12 08:43:45 : Protocol error 'invalid netstring length'
    2013-12-12 08:43:45,967 scgi-wsgi ERROR    Protocol error 'invalid netstring length'
    2013-12-12 08:44:05 : Protocol error 'invalid netstring length'
    2013-12-12 08:44:05,194 scgi-wsgi ERROR    Protocol error 'invalid netstring length'

::

    [blyth@belle7 site-packages]$ pwd
    /data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages
    [blyth@belle7 site-packages]$ unzip -p flup-1.0.3.dev_20110405-py2.5.egg flup/server/scgi_base.py  | more

::


    def readNetstring(sock):
        # Attempt to read a netstring from a socket.
        # First attempt to read the length.
        size = ''

        while True:
            try:
                c = sock.recv(1)
            except socket.error, e:
                if e[0] == errno.EAGAIN:
                    select.select([sock], [], [])
                    continue
                else:
                    raise
            if c == ':':
                break
            if not c:
                raise EOFError
            size += c

        # Try to decode the length.
        try:
            size = int(size)
            if size < 0:
                raise ValueError
        except ValueError:
            raise ProtocolError, 'invalid netstring length'

        # Now read the string.
        s, length = recvall(sock, size)


* https://github.com/joelburget/benchmarks/blob/master/flup/server/scgi_base.py



"""
import os, sys, logging, time
log = logging.getLogger(__name__)
import web
from env.geant4.geometry.collada.g4daenode import DAENode, getSubCollada, DAESubTree, getTextTree

opts = None

class Defaults(object):
    """
    """
    logpath = None
    loglevel = "INFO"
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    daepath = "$DAE_NAME_DYB"
    port = "8080"
    webpy = "scgi"
    uploads = False
    uploaddir = "$LOCAL_BASE/env/geant4/geometry/g4daeserver/samples"

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )
    op.add_option("-p", "--daepath", default=defopts.daepath , help="Path to the original geometry file. Default %default ")
    op.add_option(      "--uploads", action="store_true", default=defopts.uploads , help="Allow receiving uploaded PNGs. Default %default ")
    op.add_option(      "--uploaddir", default=defopts.uploaddir , help="Directory to store uploaded PNGs. Default %default ")
    op.add_option(      "--port",  default=defopts.port , help="Webserving port passed to webpy. Default %default ")
    op.add_option("-w", "--webpy", default=defopts.webpy , help="Webserving protocal config argv passed to webpy eg scgi fcgi. Default %default ")

    opts, args = op.parse_args()
    webpyarg = "127.0.0.1:%s %s" % ( opts.port, opts.webpy ) 
    sys.argv[1:] = webpyarg.split()   # set argv for webpy 

    level = getattr( logging, opts.loglevel.upper() )
    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))
    daepath = os.path.expandvars(os.path.expanduser(opts.daepath))
    if not daepath[0] == '/':
        daepath = os.path.abspath(daepath)
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, create the directory and move the .dae  there")
    opts.daepath = daepath
    return opts, args


cache = False
webglbook_render = web.template.render(os.path.join(os.path.dirname(__file__),'templates/webglbook'),cache=cache)
r62_render = web.template.render(os.path.join(os.path.dirname(__file__),'templates/r62'),cache=cache)

class _index:
    def GET(self):
        return "klop"

class _tree_dae:
    def GET(self, arg):
        web.header('Content-Type', 'model/vnd.collada+xml' )  # IANA mime type for DAE
        return getSubCollada(arg, dict(web.input().items()))
class _tree_txt:
    def GET(self, arg):
        log.info("_tree_txt %s " % repr(arg))
        return getTextTree(arg, dict(web.input().items())) 


def parse_getarg( arg ):
    elem = arg.split("___") 
    if len(elem) == 2:
        nodespec, maxdepth = elem
    elif len(elem) == 1:  
        nodespec = elem[0]
        maxdepth = -1
    else:
        raise Exception("failed to parse GET argument %s " % arg )
    pass 
    return nodespec, maxdepth
        
class _tree_webglbook_html:
    def GET(self, arg):
        nodespec, maxdepth = parse_getarg( arg )
        node = DAENode.get(nodespec)
        subtree = DAESubTree( node, maxdepth=maxdepth, text=False )
        return webglbook_render.production_loader_collada(arg, node, subtree )

class _tree_r62_html:
    def GET(self, arg):
        """
        #cachekiller = "?cachekiller=%s" % time.time()   
        #    Argh this kills Safari JS debugging too, will not stop on breakpoints 
        """
        nodespec, maxdepth = parse_getarg( arg )
        node = DAENode.get(nodespec)
        subtree = DAESubTree( node, maxdepth=maxdepth, text=False )
        cachekiller = ""  
        return r62_render.daeload(arg, node, subtree, cachekiller )


def find_unique_path( dir, name):
    names = os.listdir(dir)
    if not name in names:
        return os.path.join(dir, name)  
    base, ext = os.path.splitext(name) 
    counter = 1 
    uname = name
    while uname in names:
        uname = base + "_%0.2d" % counter + ext
        counter += 1
    return os.path.join(dir, uname)    

class _tree_png:
    def PUT(self, name):
        """
        ::

            curl -T snapshot05.png http://belle7.nuu.edu.tw/g4dae/tree/
            curl -T "snapshot0[0-5].png" http://belle7.nuu.edu.tw/g4dae/tree/    
                # curl needs to do the globbing not the shell, so need the quotes

        For large files (>1M) getting "413: Request Entity too large" 
        from nginx until increase the limit with the below inside the 
        http block of nginx config::

              client_max_body_size 2M;
       
        Currently samples just listed by nginx at
 
        * http://belle7.nuu.edu.tw/samples/

        To do something more could use X-Sendfile approach, together with 
        a dynamic listing.

        * http://wiki.nginx.org/X-accel
        * http://wiki.nginx.org/XSendfile
 
        """
        if not opts.uploads:
            return "uploading is disabled, restart server with `--uploads` option to allow " 
        log.info("PUT received %s  " % (name) )
        updir = os.path.expandvars(opts.uploaddir) 
        if name.startswith("..") or name.find("/") > -1:
            log.info("PUT request with disallowed resource name %s " % name )
            return None
        if not os.path.exists(updir):
            log.info("upload dir does not exist  %s " % updir )
            return None
        pass
        data = web.data()
        log.info("PUT data received %s : %s bytes " % (name,len(data)) )
        path = find_unique_path( updir, name )     
        fp = open(path, "wb")
        fp.write(data)
        fp.close()
        log.info("PUT %s wrote %s bytes to %s  " % (name,len(data), path) )
        return "OK"


URLS = (
          '/',                              '_index', 
          '/tree/(.+)?.webglbook.html',     '_tree_webglbook_html',
          '/tree/(.+)?.html',               '_tree_r62_html',
          '/tree/(.+)?.dae',                '_tree_dae',
          '/tree/(.+)?.txt',                '_tree_txt',
          '/tree/(.+\.png)',                '_tree_png',
       )

def main():
    global opts
    opts, args = parse_args(__doc__) 
    log.info("g4daeserver startup with webpy %s " % web.__version__ ) 
    DAENode.parse( opts.daepath )
    app = web.application(URLS, globals() ) 
    app.run() 

if __name__ == "__main__":
    main()


