#!/usr/bin/env python
"""
DAEDIFF
========

Document the differences between the orignal .dae and the 
vnode.py subcopy (when recursively subcopying everything ie from PV/LV top.0/World)


"""
import sys, os, logging, hashlib
log = logging.getLogger(__name__)
from pprint import pformat
import lxml.etree as ET
import numpy
parser = ET.XMLParser(remove_blank_text=True)
parse_ = lambda _:ET.parse(os.path.expandvars(_),parser=parser).getroot()
tostring_ = lambda _:ET.tostring(_, pretty_print=True)
id_ = lambda _:_.attrib.get('id',None)

COLLADA_NS='http://www.collada.org/2005/11/COLLADASchema'
tag_ = lambda _:"{%s}%s" % ( COLLADA_NS, _ )
xpath_ = lambda _:"./"+"/".join(map(tag_,_.split("/"))) 
stag_ = lambda _:_[len(COLLADA_NS)+2:]
xfind_ = lambda elem,_:elem.find(xpath_(_))

nswap = 0 


def normalize_geometry( elem ):
    """
    lxml attribute ordering

    * https://pypi.python.org/pypi/lxml/3.0

    Element attributes passed in as an attrib dict or as keyword arguments are
    now sorted by (namespaced) name before being created to make their order
    predictable for serialisation and iteration. Note that adding or deleting
    attributes afterwards does not take that order into account, i.e. setting a
    new attribute appends it after the existing ones.  

    I have ET.LXML_VERSION (2, 3, 0, 0)
    
    """
    # use common float handling to prevent precision/notation differences
    for fa in elem.findall(".//{%s}float_array" % COLLADA_NS ):
        a = numpy.fromstring(fa.text, dtype=numpy.float32, sep=' ')   
        fa.text = " ".join(map(str,a.tolist()))

    # normalise the type/name order of the attributes on <param type="float" name="Y"/>
    for p in elem.findall(".//"+tag_("param")):
        p.set('type', p.attrib.pop('type'))   # places type second after name for all param

    s = tostring_(elem)
    s = s.replace("\n\t\t\t\t","")
    s = s.replace("\n","")
    return s

def normalize_material( elem ):
    if 'name' in elem.attrib:
        del elem.attrib['name']
    return tostring_(elem)

def normalize_effect( elem ):
    if 'name' in elem.attrib:
        del elem.attrib['name']

    # an incorrect sid is set in original it seems
    tech = elem.find(".//{%s}technique" % COLLADA_NS ) 
    if tech is not None:
        tech.set('sid','common')   

    # get rid of an extra added by pycollada creation '<extra><technique profile="GOOGLEEARTH"><double_sided>0</double_sided></technique></extra>'
    pc =  elem.find(".//{%s}profile_COMMON" % COLLADA_NS )
    if pc is not None:
        if len(pc) == 2 and pc[1].tag == tag_("extra"):
            pc.remove(pc[1])

    s = tostring_(elem)
    # kludge the color numbers difference, 1 1 1 1 in orig becomes 1.0 1.0 1.0 1.0 in the copy 
    s = s.replace(".0","")   
    return s
 
def normalize_node( elem ):
    """
    sometimes the <matrix> and <instance_node> element ordering is swapped in the copy 
    for unknown reasons

    Simple handling of 2 elements will not work as the elem can be a compound node
    """
    matrix = elem.find("{%s}matrix" % COLLADA_NS )
    instance_node = elem.find("{%s}instance_node" % COLLADA_NS )
    if instance_node is not None and matrix is not None:
        s = tostring_(instance_node) + tostring_(matrix)
    else:    
        s = tostring_(elem)
    return s    

def rswap( xn , zero=False ):
    """
    Recursively fix the matrix/instance_node swappage, acting directly on the etree.
    Subsequent application does nothing.

    **NB** This only works with lxml.etree  http://lxml.de/tutorial.html
    """
    global nswap 
    if zero:
        nswap = 0
    if len(xn) == 2 and xn[0].tag == tag_("instance_node") and xn[1].tag == tag_("matrix"):
        xn.insert(0,xn[1])
        nswap += 1
    else:
        for _ in xn:
            rswap(_)


def normalize(elem):
    """
    Changes to bring the orig and copy closer together, in order to identify differnces
    """
    if tag_("effect") == elem.tag:
         s = normalize_effect(elem)
    elif tag_("material") == elem.tag:
         s = normalize_material(elem)
    elif tag_("geometry") == elem.tag:
         s = normalize_geometry(elem)
    elif tag_("node") == elem.tag:
         s = normalize_node(elem)
    else: 
         s = tostring_(elem)
    return s




class DAEDiff(object):
    def __init__(self, orig, copy ):
        log.info("daediff orig %s copy %s " % (orig, copy ))    
        orig, copy  = map(parse_, [orig, copy])

        ocreated = xfind_(orig,"asset/created").text
        ccreated = xfind_(copy,"asset/created").text
        print "created o %s c %s  " % (ocreated, ccreated)
        otops = orig.findall("./*")
        ctops = copy.findall("./*")
        tld = self.toplevel(otops, ctops)
        pass

        # recursively diddle the etree, fix instance_node/matrix element order swaps to be in expected matrix first order 
        rswap(copy, zero=True)   
        log.info("swapped %s in copy " % nswap )
        rswap(orig, zero=True)   
        log.info("swapped %s in orig " % nswap )

        self.orig = orig 
        self.copy = copy 
        self.tld = tld 

    def toplevel(self, otops, ctops):
        df = {}
        for otop,ctop in zip(otops, ctops):
            assert otop.tag == ctop.tag, (otop, ctop)
            tag = stag_(otop.tag)
            if tag not in df:
                df[tag] = {}
                df[tag]['element'] = {}
            pass    
            df[tag]['element']['orig'] = otop 
            df[tag]['element']['copy'] = ctop 
            pass
        print pformat(df)
        return df 

    def sublevel(self, name):
        """
        To handle different element ordering, collect into a dict 
        keyed on the id 
        """
        cf = {}
        for type in "orig copy".split():
            for elem in self.tld[name]['element'][type]:
                id = elem.attrib['id']
                if id not in cf:
                    cf[id] = {} 

                cf[id][type] = normalize(elem)
            pass    

        ndif = 0 
        for id in cf:
            o = cf[id]['orig']
            c = cf[id]['copy']
            same = o == c 
            cf[id]['same'] = same
            if not same:
                ndif += 1
        return cf, ndif




def diffstr( o, c, cut=1000):
    """
    """
    m = min(len(o),len(c))
    print "o %s c %s m %s\n" % (len(o), len(c), m)
    if m < cut:
        print "o %s\n%s" % (len(o),o)
        print "c %s\n%s" % (len(c),c)
    else:
        # bychar handling for comparing strings too large to compare by eye
        x = range(0,m)
        w = 10 
        d = filter(lambda _:o[_] != c[_], x )  # char indices that are different
        print "# diff char %s first %s " % ( len(d), d[0] )
        print "o " + "".join([o[_] for _ in range(d[0]-w, d[0]+w)])
        print "c " + "".join([c[_] for _ in range(d[0]-w, d[0]+w)])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    orig = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
    if len(sys.argv)>1:
        copy = sys.argv[1]
    else:    
        copy = "0.xml"

    dd = DAEDiff( orig, copy )

    dflib = "library_materials library_effects library_geometries library_nodes"
    #dflib = "library_nodes"
    #dflib = "library_effects"
    #dflib = "library_geometries"

    for lib in dflib.split():
        cf,nd = dd.sublevel(lib)
        print 
        print lib, nd 
        if nd > 0:
            for id in cf.keys()[-10:]:
                if not cf[id]['same']:
                    print
                    print id, cf[id]['same']
                    o = cf[id]['orig']
                    c = cf[id]['copy']
                    diffstr(o,c)









    


