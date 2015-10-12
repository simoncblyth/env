#!/usr/bin/env python
"""
::

    In [40]: hemi.findall_("./*")
    Out[40]: 
    [          union : {'name': 'pmt-hemi'} ,
             physvol : {'logvol': '/dd/Geometry/PMT/lvPmtHemiVacuum', 'name': 'pvPmtHemiVacuum'} ]




"""



import os, re, logging
import lxml.etree as ET
import lxml.html as HT
try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None


tostring_ = lambda _:ET.tostring(_)
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())

isparameter_ = lambda _:_.tag == 'parameter'
islogvol_ = lambda _:_.tag == 'logvol'
issphere_ = lambda _:_.tag == 'sphere'


log = logging.getLogger(__name__)

class Att(object):
    leaf = False 
    op_expr = re.compile("([\+-])")
    simple_expr = re.compile("(\w*)")
    binary_expr = re.compile("(\w*)([\+-])(\w*)")

    def __init__(self, expr, g):
        self.expr = expr
        self.g = g  

    def __repr__(self):
        return "%s : %s " % (self.expr, self.eval_(self.expr))

    def match(self, expr):
        self.m_simple = self.simple_expr.match(expr)
        self.m_op = self.op_expr.match(expr)
        self.m_binary = self.binary_expr.match(expr)

      

    def eval_(self, expr):
        if self.leaf:
            return [expr]

        self.match(expr)
        ret = []
        if self.m_op:
            log.debug("m_op %s " % expr)
            ret.extend([self.m_op.groups()[0]]) 
        elif self.m_binary:
            log.debug("m_binary %s " % expr)
            for _ in self.m_binary.groups():
                log.debug("m_binary %s " % _ )
                ret.extend( self.eval_(_))
            pass
        elif self.m_simple:
            log.debug("m_simple %s " % expr)
            k = self.m_simple.groups()[0]
            v = self.g.params.get(k,'?')
            ret.extend(["%s:%s" % (k,v)])
        else:
            log.info("skipped %s " % expr)
            ret.extend(["%s:%s" % (expr,"?")])
            pass

        log.debug("eval_ %s  -> %s " % (expr, repr(ret)))
        return ret 


class LeafAtt(Att):
    leaf = True




class Elem(object):
    def __init__(self, elem, g=None):
        self.elem = elem 
        self.g = g 

    def att(self, k, leaf=False):
        v = self.elem.attrib.get(k, None)
        kls = LeafAtt if leaf else Att 
        return kls(v, self.g) if v is not None else None 

    def findall_(self, expr):
        return map( lambda e:self.g.kls.get(e.tag,Elem)(e,self.g), self.elem.findall(expr) )

    def find_(self, expr):
        e = self.elem.find(expr) 
        return self.g.kls.get(e.tag,Elem)(e,self.g) if e is not None else None

    def __repr__(self):
        return "%15s : %s " % ( self.elem.tag, repr(self.elem.attrib) )






class Parameter(Elem):
    name  = property(lambda self:self.elem.attrib['name'])
    value = property(lambda self:self.elem.attrib['value'])

    def hasprefix(self, prefix):
        return self.name.startswith(prefix)

    def __repr__(self):
        a = self.elem.attrib
        return "%30s : %s " % ( a['name'], a['value'] )

class Logvol(Elem):
    def __repr__(self):
        a = self.elem.attrib
        return "%30s %20s %s " % (a['name'], a.get('material',"-"), a.get("sensdet","-"))

class Sphere(Elem):
    name = property(lambda self:self.att('name', leaf=True))
    outerRadius = property(lambda self:self.att('outerRadius'))
    def __repr__(self):
        return "sphere %20s : %s  " % (self.name, self.outerRadius)

class PosXYZ(Elem):
    z = property(lambda self:self.att('z'))
    def __repr__(self):
        return "PosXYZ  %s  " % (repr(self.z))




class Dddb(Elem):
    kls = {
        "parameter":Parameter,
        "sphere":Sphere,
        "logvol":Logvol,
        "posXYZ":PosXYZ,
    }

    @classmethod
    def parse(cls, path):
        g = Dddb(parse_(path))
        g.init()
        return g

    def init(self):
        self.g = self
        self.params = {}
        for p in self.params_():
            self.params[p.name] = p.value


    def logvol_(self, name):
        return self.find_(".//logvol[@name='%s']"%name)

    def logvols_(self):
        return self.findall_(".//logvol")

    def params_(self, prefix=None):
        pp = self.findall_(".//parameter")
        if prefix is not None:
            pp = filter(lambda p:p.hasprefix(prefix), pp)
        return pp  




def parameters(path="$PMT_DIR/hemi-parameters.xml"):
   hp = filter(isparameter_,fparse_(path)) 
   return hp


def circle_intersect( a, b ):
   """
   http://mathworld.wolfram.com/Circle-CircleIntersection.html
   """
   R = a.radius
   r = b.radius

   xy_a = a.center
   xy_b = b.center

   log.info(" A %s xy %s " % ( R, repr(xy_a)) )  
   log.info(" B %s xy %s " % ( r, repr(xy_b)) )  

   assert(xy_b[1] == xy_a[1]) 

   d = xy_b[0] - xy_a[0]

   x = (d*d - r*r + R*R)/(2.*d)

   return x + xy_a[0] 


def pmt_hemi(p):

    xy1 = (0.,0.)
    xy2 = (p['PmtHemiFaceOff-PmtHemiBellyOff'],0.)
    xy3 = (p['PmtHemiFaceOff+PmtHemiBellyOff'],0.)

    r1 = p['PmtHemiFaceROC']
    r2 = p['PmtHemiBellyROC']
    r3 = p['PmtHemiBellyROC']
 
    c1=plt.Circle(xy1,r1, edgecolor='r', facecolor='w', alpha=0.5)
    c2=plt.Circle(xy2,r2, edgecolor='g', facecolor='w', alpha=0.5)
    c3=plt.Circle(xy3,r3, edgecolor='b', facecolor='w', alpha=0.5)
  
    c12 = circle_intersect(c1,c2)
    c13 = circle_intersect(c1,c3)
    c23 = circle_intersect(c2,c3)

    xl = (-200,200)
    yl = (-200,200)
   
    nl = 3 
    l1 = [[c12,c12],[c13,c13],[c23,c23]]
    l2 = [yl,yl,yl]

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches


    bxy = [p['-0.5*PmtHemiGlassBaseLength'],-1.*p['PmtHemiGlassBaseRadius']]
    bwh = [p['PmtHemiGlassBaseLength'], 2.0*p['PmtHemiGlassBaseRadius']]   
 
    t1 = plt.Rectangle(bxy, bwh[0], bwh[1], edgecolor='r', facecolor='w', alpha=0.5) 

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

    ax.add_artist(c1)
    ax.add_artist(c2)
    ax.add_artist(c3)
    ax.add_artist(t1)

    for i in range(nl):
        a = 0.1 if i == 1 else 0.9
        line = mlines.Line2D( l1[i], l2[i] , lw=5., alpha=a)
        ax.add_line(line)

    ax.autoscale_view(True,True,True)
    fig.show()

    fig.savefig('/tmp/pmt_hemi.png')



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")

    #print "\n".join(map(repr, g.logvols_()))
    print "\n".join(map(repr, g.params_('PmtHemi')))

    hemi = g.logvol_('lvPmtHemi')

    for _ in hemi.findall_("./union/*"):
        print _


    # NB parsin still manually done

    p = {}
    p['PmtHemiFaceROC'] = 131.
    p['PmtHemiBellyROC'] = 102.
    p['PmtHemiFaceOff'] = 56.
    p['PmtHemiBellyOff'] = 13.

    p['300*mm-PmtHemiFaceROC'] = 300.-131.
    p['42.25*mm'] = 42.25

    p['PmtHemiGlassBaseLength'] = p['300*mm-PmtHemiFaceROC'] 
    p['PmtHemiGlassBaseRadius'] = p['42.25*mm']
    p['-0.5*PmtHemiGlassBaseLength'] = -0.5*p['PmtHemiGlassBaseLength']

    p['PmtHemiFaceOff-PmtHemiBellyOff'] = p['PmtHemiFaceOff'] - p['PmtHemiBellyOff']
    p['PmtHemiFaceOff+PmtHemiBellyOff'] = p['PmtHemiFaceOff'] + p['PmtHemiBellyOff']

    pmt_hemi(p)


