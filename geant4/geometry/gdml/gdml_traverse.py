#!/usr/bin/env python
import os, logging, re, sys
from env.db.simtab import Table
log = logging.getLogger(__name__)

ptr = re.compile("0x[0-9a-fA-F]{7}$")

#from xml.etree import ElementTree as ET
import xml.etree.cElementTree as ET

class Parse(dict):
    def __init__(self, path, opts):
        dict.__init__(self)
        self.path = path
        self.opts = opts
        self.parse_gdml()

    def __call__(self):
        dbpath = self.prepdb()
        self.dbpath = dbpath
        self.recurse(self.structure, self.physvol_)
        self.recurse(self.structure, self.volume_ )
        self.savedb()

    def prepdb(self):    
        path = os.path.abspath(self.path)
        base, ext = os.path.splitext(path)
        dbpath = base + ".gdml.db"
        if os.path.exists(dbpath):
            log.info("remove pre-existing db file %s " % dbpath)
            os.remove(dbpath)
        pass
        self['physvol_names'] = [] 
        self['physvol_table'] = Table(dbpath, "physvol", id="int",name="text", uname="text" )
        self['volume_names'] = [] 
        self['volume_table'] = Table(dbpath, "volume", id="int",name="text", uname="text" )
        return dbpath

    def savedb(self):
        log.info("start persisting to %s " % self.dbpath ) 
        self['physvol_table'].insert()
        self['volume_table'].insert()
        log.info("completed persisting to %s " % self.dbpath ) 

    def volume_(self, node):
        if node.tag != 'volume':return
        self.collect_name( node.attrib['name'], 'volume_')

    def physvol_(self, node):
        if node.tag != 'physvol':return
        self.collect_name( node.attrib['name'], 'physvol_')

    def print_(self, node):
        print node.tag, node.attrib

    def collect_name(self, uname, type):
        assert ptr.search(uname), "expecting last 9 chars of name to look like a pointer eg ..CrossRib10xc4f2a08  0xc4f2a08 "
        name = uname[:-9]
        self[type+'names'].append(name)
        id = self.opts.idoffset + len(self[type+'names'])
        self[type+'table'].add(id=id, name=name, uname=uname)

    def recurse(self, base, fn ):
        fn( base )
        for node in base:
            self.recurse(node, fn)

    def dump(self):
        for line in file(self.path).readlines():
            print line,

    def parse_gdml(self):    
        log.info("parsing %s " % self.path ) 
        root = ET.parse(self.path).getroot()
        log.info("completed parse")
        self.root = root 
        pass
        define, materials, solids, structure, setup = tuple([_ for _ in root])
        assert define.tag == 'define'
        assert materials.tag == 'materials'
        assert solids.tag == 'solids'
        assert structure.tag == 'structure'
        assert setup.tag == 'setup'
        pass 
        self.define = define
        self.materials = materials
        self.solids = solids
        self.structure = structure
        self.setup = setup



class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    idoffset = 0 
    gdmlpath = '$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml'   # truncation lifted

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat )
    op.add_option("-i", "--idoffset", type=int, default=defopts.idoffset)
    op.add_option("-g", "--gdmlpath", default=defopts.gdmlpath )

    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()              # py2.3 has unusable basicConfig that takes no arguments
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass
    log.info(" ".join(sys.argv))
    gdmlpath = os.path.expandvars(os.path.expanduser(opts.gdmlpath))
    if not gdmlpath[0] == '/':
        opts.gdmlpath = os.path.join(os.path.dirname(__file__),gdmlpath)
    else:
        opts.gdmlpath = gdmlpath 

    base, ext = os.path.splitext(os.path.abspath(gdmlpath))
    dbpath = base + ".gdml.db"
    opts.dbpath = dbpath
    assert os.path.exists(gdmlpath), (gdmlpath,"GDML file not at the new expected location, please create the directory and move the .gdml  there, please")
    pass    
    return opts, args


if __name__ == '__main__':
    opts, args = parse_args(__doc__) 
    pr = Parse(opts.gdmlpath, opts)
    pr()


