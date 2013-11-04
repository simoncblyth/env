#!/usr/bin/env python
"""

Usage::

   daeserver.py --help
   daeserver.py                              # scgi/apache on G 
   daeserver.py -w "127.0.0.1:8080 fcgi"     # fastcgi/nginx on N 

   daeserver.py --daepath $LOCAL_BASE/env/graphics/collada/3199.dae  
      # starting from a small DAE file is convenient for parsing speed during development
      # NB volume indices will then be relative to that sub-root  

With webpy SCGI deployment with apache

* http://localhost/dae/hello/hello.html?name=simon
* http://localhost/dae/tree/0___0.html
* http://localhost/dae/tree/3199___0.html
* http://localhost/dae/tree/0___0.dae


"""
import os, sys, logging
log = logging.getLogger(__name__)
import web
from env.graphics.collada.pycollada.daenode import DAENode, getSubCollada, DAESubTree


class Defaults(object):
    logpath = None
    loglevel = "INFO"
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
    webpy = "127.0.0.1:8080 scgi"

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )
    op.add_option("-p", "--daepath", default=defopts.daepath , help="Path to the original geometry file. Default %default ")
    op.add_option("-w", "--webpy", default=defopts.webpy , help="Webserving config argv passed to webpy. Default %default ")

    opts, args = op.parse_args()
    sys.argv[1:] = opts.webpy.split()   # set argv for webpy 

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
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    opts.daepath = daepath
    return opts, args


webglbook_render = web.template.render(os.path.join(os.path.dirname(__file__),'templates/webglbook'))
r62_render = web.template.render(os.path.join(os.path.dirname(__file__),'templates/r62'))

class _index:
    def GET(self):
        return "klop"

class _tree_dae:
    def GET(self, arg):
        return getSubCollada(arg, dict(web.input().items()))

class _tree_html:
    def GET(self, arg):
        if '___' in arg: 
            maxdepth = arg.split("___")[1]
        else:
            maxdepth = -1
        node = DAENode.get(arg)
        subtree = DAESubTree( node, maxdepth=maxdepth, text=False )
        return webglbook_render.production_loader_collada(arg, node, subtree )

class _tree_htm:
    def GET(self, arg):
        if '___' in arg: 
            maxdepth = arg.split("___")[1]
        else:
            maxdepth = -1
        node = DAENode.get(arg)
        subtree = DAESubTree( node, maxdepth=maxdepth, text=False )
        return r62_render.daeload(arg, node, subtree )


URLS = (
          '/',                    '_index', 
          '/tree/(.+)?.html',     '_tree_html',
          '/tree/(.+)?.htm',      '_tree_htm',
          '/tree/(.+)?.dae',      '_tree_dae',
       )

def main():
    opts, args = parse_args(__doc__) 
    DAENode.parse( opts.daepath )
    app = web.application(URLS, globals())
    app.run() 

if __name__ == "__main__":
    main()


