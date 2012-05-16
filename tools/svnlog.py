#!/usr/bin/env python
"""
This script queries the commit log of the SVN repository
corresponding to the working copy of the invoking directory.
Invoking from outside SVN working copy results in an error.

Usage examples:: 

    svnlog -w 52 -a blyth     ## dump 52 weeks of commit messages for single author

    svnlog --limit 1000000 -w 52 -a blyth > 2012.txt    ## up the limit to avoid truncation

    svnlog --limit 1000000 -v debug -a blyth  
    svnlog --limit 1000000 -v debug -a blyth > ~/2011.txt


NB during development, duplicate arguments precisely to benefit from caching 

"""
import re, os, logging, md5
from xml.dom import minidom 
from dt import DT
from datetime import timedelta
from misc import getuser 

log = logging.getLogger(__name__)

class Node(list):
    """
    Base class for wrapping python around an xml emitting command 
    such as  `svn info --xml` or `svn log --xml`
    """
    _cmd = "<msg>implement xml producing cmd in subclass %(var)s </msg>"

    def _src(self, ctx ):
        """
        :param ctx: 

        cached call with keyed on the command
        """ 

        base = ctx.get('base','.')
        if not base.startswith('http'):
            ctx['base'] = os.path.abspath(base)    ## because "." is just too relative 

        _cmd = self._cmd % ctx 
        dig = md5.new(_cmd).hexdigest() 
        xmlcache = os.path.join( "/tmp/%s/env/tools/svnlog" % getuser() , "%s.xmlcache" % dig )
        log.debug("_cmd %s  " % _cmd )        
        log.debug("xmlcache %s " % xmlcache )
        if os.path.exists(xmlcache):
            log.warn("reading from xmlcache %s " % xmlcache) 
        else:  
            dir = os.path.dirname(xmlcache)
            if not os.path.exists(dir):
                 os.makedirs( dir ) 
            _cmd += " > %s " % xmlcache  
            ret = os.popen(_cmd).read()
            log.info("writing to xmlcache %s %s " % (xmlcache, ret ))
        return file(xmlcache,"r").read()

    def _get_one(self, tag):
        lec = self.node.getElementsByTagName(tag)
        if len(lec) == 1:
            if lec[0].firstChild:  ## special handling for empty elements
                return lec[0].firstChild.data       
        else:
            log.warn("getElementsByTagName unexpected lec %s %s " % (lec,tag) )
            return None 
    def _get_data(self):
        return self.node.firstChild.data
    def _get_att(self, att):
        return self.node.getAttribute(att)
    def __init__(self, node=None, parse=None):
        if parse:
            node = minidom.parseString( parse ) 
        self.parent = None
        self.node = node 
    def _rootnode(self):
        if not self.parent:
             return self
        else:
             return self.parent.rootnode
    rootnode = property( _rootnode )  

    def __call__(self, cls, tag ):
        """
        :param cls: class to represent child elements in the xml document, the class must implement
                    a single argument ctor that takes a mindom parsed instance
        :param tag: name of the xml child element

        Child element nodes are "wrapped" by the `cls` and  appended to this list
        """
        for e in self.node.getElementsByTagName(tag):
            c = cls(e)
            c.parent = self 
            self.append(c)
    def __str__(self):
        return self.node.toprettyxml()
    def __repr__(self):
        return "\n".join([repr(c) for c in self])


class Path(Node):
    action = property( lambda self:self._get_att("action") )
    path = property( lambda self:self._get_data())
    def __init__(self, node):
        Node.__init__(self, node)
    def __repr__(self):
        return "   %s %s"  % ( self.action, self.path )

class LogEntry(Node):
    revision = property( lambda self:self._get_att("revision") )
    author = property( lambda self:self._get_one("author") )
    date   = property( lambda self:self._get_one("date") )
    t = property( lambda self:DT(self.date).t )
    age = property( lambda self:self.parent.t - self.t )
    msg    = property( lambda self:self._get_one("msg") or "naughty author ... no commit message" )
    selected = property( lambda self:self.age < self.parent.maxage )
    sauthor = property( lambda self:self.author == self.parent.author )

    def __init__(self, node):
        Node.__init__(self, node)
        self(Path, "path")

    def __repr__(self):
        ok = ["","**"][self.selected] 
        if self.rootnode.verbose: 
            v = "\n".join( [repr(c) for c in self]) + "\n"
        else:
            v = ""
        return "%s %s %s %s %s \n    %s\n" % ( ok, self.age, self.revision, self.author, self.date, self.msg ) + v


class Msg(str):
    """
    A labelled string, usage::

        m = Msg('hello reref:pkg')(age=60)
        assert m == 'hello reref:pkg'
        assert m.age == 60 
 
    """
    def __call__(self, **kwa):
        for k,v in kwa.items():
            setattr( self, k, v )
        return self


class SVNLog(Node):
    t = property( lambda self:self[0].t )
    _cmd = "svn log --limit %(limit)s --verbose --xml --revision %(revision)s %(base)s "
    def __init__(self, base, revision, opts , maxage=timedelta(0,60*60), verbose=False ):
        log.info("SVNLog base %s " % base )
        if opts.get('revision',None):
            log.warn("option override of revision %s to %s ", revision, opts['revision'] )
            revision = opts['revision']  
        self.ctx = dict(base=base, revision=revision, limit=int(opts['limit']) )
        self.maxage = maxage
        self.author = opts.get('author', None) 
        self.verbose = verbose
        Node.__init__(self, parse=self._src(self.ctx) )
        self(LogEntry, "logentry")
    
    def selection(self, predicate=lambda c:c.selected):
        return filter( predicate , self )

    def writelog(self):
        for c in self.selection():
            log.info("%r", c ) 
    def msgs(self):
        _msgs = []
        for le in self.selection():
            msg = Msg(le.msg)(age=le.age,logentry=le)
            _msgs.append(msg)
        return _msgs 
    def __repr__(self):
        return "\n".join([repr(c) for c in self.selection()])



class Repository(Node):
    root = property( lambda self:self._get_one("root") )
    uuid = property( lambda self:self._get_one("uuid") )
    def __init__(self, node):
        Node.__init__(self, node )
    def __repr__(self):
        return " %s %s " % ( self.root, self.uuid )

class Info(Node):
    """
    Used to determine the root url of the SVN repository corresponding
    to the current working directory. 

    Will fail if not invoked from svn working copy directory.
    """
    _cmd = "svn info --xml %(base)s "

    def __init__(self, base="." ):
        Node.__init__(self, parse=self._src(dict(base=base)) )
        self(Repository, "repository")

        self.rooturl = self[0].root 
        assert self.rooturl.startswith('http')
        entries = self.node.getElementsByTagName("entry")
        assert len(entries)==1, entries
        self.path     = entries[0].getAttribute("path")
        self.revision = entries[0].getAttribute("revision")

    def __repr__(self):
        return "Info %s %s %s " % ( self.rooturl, self.path, self.revision )



class StatusEntry(Node):
    """
    <entry  path="/home/blyth/dybaux/catalog/tmp_offline_db/CableMap/CableMapVld.csv">
      <wc-status props="none" item="modified" revision="4915">
        <commit revision="4915">
          <author>bv</author>
          <date>2011-06-21T16:48:04.006560Z</date>
        </commit>
      </wc-status>
    </entry>
    """
    def __init__(self, node):
        Node.__init__(self, node )
        wcstatus = self.node.getElementsByTagName("wc-status")
        assert len(wcstatus) == 1, wcstatus
        self.path = self._get_att("path") 
        self.status   = wcstatus[0].getAttribute("item") 
        self.revision = wcstatus[0].getAttribute("revision") 

    def __repr__(self):
        return "%s %s %s %s" % ( self.__class__.__name__ , self.path, self.status, self.revision )


class Status(Node):
    """
<?xml version="1.0"?>
<status>
  <target path="/home/blyth/dybaux/catalog/tmp_offline_db">
    <entry  path="/home/blyth/dybaux/catalog/tmp_offline_db/CableMap/CableMapVld.csv">
      <wc-status props="none" item="modified" revision="4915">
        <commit revision="4915">
          <author>bv</author>
          <date>2011-06-21T16:48:04.006560Z</date>
        </commit>
      </wc-status>
    </entry>
  </target>
</status>
    """ 
    _cmd = "svn status --xml %(base)s "
    def __init__(self, base="." ):
        Node.__init__(self, parse=self._src(dict(base=base)) )

        targets = self.node.getElementsByTagName("target")
        assert len(targets) == 1, targets
        self.target = targets[0].getAttribute("path")
        self(StatusEntry, "entry")
    def __repr__(self):
        return "%s %s" % ( self.__class__.__name__ , self.target )

    



class Msgs(list):
    """
    `svn info --xml` is parsed to get the repository root which is then used
    in `svn log --xml` in order to collect the last 30 log entries

    Only log entries within 60 mins of the last one are selected
    and collected into this list of commit messages. 

    """
    defaults = dict(loglevel="INFO", limit="30", base="." , weeks="52", author=None )

    def parse_args( cls ):
        from optparse import OptionParser
        op = OptionParser(usage=__doc__)
        d = cls.defaults
        op.add_option("-l", "--limit" ,     help="limit number of revisions to look at. Default %(limit)s " % d  )
        op.add_option("-r", "--revision" ,  help="OVERRIDE auto determined revision sequence, FOR DEBUGGING ONLY eg use 10891:1  "  )
        op.add_option("-v", "--loglevel",   help="logging level : INFO, WARN, DEBUG ... Default %(loglevel)s " % d )
        op.add_option("-w", "--weeks" ,     help="weeks of logs to dump. Default %(weeks)s " % d )
        op.add_option("-a", "--author" ,    help="restrict selection to single author. Default %(author)s " % d )
        op.set_defaults( **cls.defaults )
        return op.parse_args()
    parse_args = classmethod( parse_args )

    def __init__(self):

        opts,args = Msgs.parse_args()
        loglevel = getattr( logging, opts.loglevel.upper() , logging.INFO )
        logging.basicConfig(level=loglevel)
        log.info("args %s opts %s " % ( repr(args), repr(vars(opts)) ) )

        maxage = timedelta(weeks=int(opts.weeks))
        info = Info()
        log.info("%r" % info )
        slog = SVNLog( info.rooturl, "%s:%s" % (info.revision,1) , vars(opts), maxage=maxage )   

        #self[:] = slog.msgs()
        self.slog = slog 

    def __repr__(self):
        return "%r" % self.slog + "\n" + "\n".join([m for m in self if m])  

  
if __name__ == '__main__':
    msgs = Msgs()
    for le in msgs.slog.selection(lambda _:_.sauthor and _.selected):
        print repr(le)
 





