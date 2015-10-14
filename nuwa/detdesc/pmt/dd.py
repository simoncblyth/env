#!/usr/bin/env python
import os, re, logging, math
import lxml.etree as ET
import lxml.html as HT
from math import acos

log = logging.getLogger(__name__)

tostring_ = lambda _:ET.tostring(_)
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])


class Att(object):
    def __init__(self, expr, g, evaluate=True):
        self.expr = expr
        self.g = g  
        self.evaluate = evaluate

    value = property(lambda self:self.g.ctx.evaluate(self.expr) if self.evaluate else self.expr)

    def __repr__(self):
        return "%s : %s " % (self.expr, self.value)


class Elem(object):
    transform = None
    is_rev = False
    name  = property(lambda self:self.elem.attrib.get('name',None))
    is_primitive = property(lambda self:type(self) in self.g.primitive)
    is_composite = property(lambda self:type(self) in self.g.composite)
    is_transform = property(lambda self:type(self) in self.g.transform)

    @classmethod
    def link_transform(cls, ls, transform):
        """
        # attach any PosXYZ instances in the list to preceeding geometry elements
        # looks like physvol can hold one too
        """
        for i in range(len(ls)):
            if ls[i].is_primitive:
                ls[i].transform = transform 
                log.debug("linking %s to %s " % (transform, ls[i]))

    @classmethod
    def link_prior_transform(cls, ls):
        """
        # attach any PosXYZ instances in the list to preceeding geometry elements
        # looks like physvol can hold one too
        """
        for i in range(len(ls)):
            if ls[i].is_transform and ls[i-1].is_primitive:
                ls[i-1].transform = ls[i] 
                log.debug("linking %s to %s " % (ls[i], ls[i-1]))

    def _get_xyz(self):
       x = y = z = 0
       if self.transform is not None:
           x = self.transform.x.value 
           y = self.transform.y.value 
           z = self.transform.z.value 
       pass
       return [x,y,z] 
    xyz = property(_get_xyz)

    def __init__(self, elem, g=None):
        self.elem = elem 
        self.g = g 

    def att(self, k, dflt=None):
        v = self.elem.attrib.get(k, None)
        return Att(v, self.g) if v is not None else Att(dflt, self.g, evaluate=False) 

    def findall_(self, expr):
        wrap_ = lambda e:self.g.kls.get(e.tag,Elem)(e,self.g)
        fa = map(wrap_, self.elem.findall(expr) )
        kln = self.__class__.__name__
        name = self.name 
        log.info("findall_ from %s:%s expr:%s returned %s " % (kln, name, expr, len(fa)))
        return fa 

    def findone_(self, expr):
        all_ = self.findall_(expr)
        assert len(all_) == 1
        return all_[0]

    def find_(self, expr):
        e = self.elem.find(expr) 
        wrap_ = lambda e:self.g.kls.get(e.tag,Elem)(e,self.g)
        return wrap_(e) if e is not None else None

    def __repr__(self):
        return "%15s : %s " % ( self.elem.tag, repr(self.elem.attrib) )

    def allrev(self, depth=0, maxdepth=10):
        if depth > maxdepth:
            return []

        if type(self) is Physvol:
            lvn = self.logvolref.split("/")[-1]
            log.info("physvol is special logvolref %s lvn %s " % (self.logvolref, lvn ))
            lv = self.g.logvol_(lvn)
            depth += 1
            components = lv.findall_("./*")  
            transform = self.get_transform()
            if transform:
                self.link_transform(components, transform)
        else:
            components = self.findall_("./*")  # one lev only
            self.link_prior_transform(components)
        pass
  

        revs = []
        for c in components:
            if c.is_primitive:
                xrev = c.asrev()
                log.debug("allrev: primitive : %s " % repr(xrev)) 
                revs.extend(xrev)
            elif c.is_composite:
                xrev = c.allrev(depth=depth+1, maxdepth=maxdepth) 
                log.debug("allrev: composite : %s " % repr(xrev)) 
                revs.extend(xrev)
            elif c.is_transform:
                pass
            else:
                log.warning("skipped component %s " % repr(c))
            pass
                  
        return revs
    
 
class Logvol(Elem):
    material = property(lambda self:self.elem.attrib.get('material', None))
    sensdet = property(lambda self:self.elem.attrib.get('sensdet', None))
    def __repr__(self):
        return "%30s %20s %s " % (self.name, self.material, self.sensdet)

class Physvol(Elem):
    logvolref = property(lambda self:self.elem.attrib.get('logvol', None))
    def __repr__(self):
        return "Physvol %20s %s " % (self.name, self.logvolref)

    def get_transform(self):
        return self.find_("./posXYZ") 


class Union(Elem):
    def __repr__(self):
        return "Union %20s  " % (self.name)

class Intersection(Elem):
    def __repr__(self):
        return "Intersection %20s  " % (self.name)


class Parameter(Elem):
    expr = property(lambda self:self.elem.attrib['value'])

    def hasprefix(self, prefix):
        return self.name.startswith(prefix)

    def __repr__(self):
        return "%30s : %s " % ( self.name, self.expr )






class Rev(object):
    def __init__(self, typ, name, xyz, radius, sizeZ=None, startTheta=None, deltaTheta=None, width=None):
        self.typ = typ
        self.name = name
        self.xyz = xyz 
        self.radius = radius
        self.sizeZ = sizeZ
        self.startTheta = startTheta
        self.deltaTheta = deltaTheta
        self.width = width

    def __repr__(self):
        return "Rev('%s','%s' xyz:%s, r:%s, sz:%s, st:%s, dt:%s wi:%s)" % \
            (self.typ, self.name, self.xyz, self.radius, self.sizeZ, self.startTheta, self.deltaTheta, self.width)

class Primitive(Elem):
    is_rev = True
    outerRadius = property(lambda self:self.att('outerRadius'))
    innerRadius = property(lambda self:self.att('innerRadius'))

class Sphere(Primitive):
    startThetaAngle = property(lambda self:self.att('startThetaAngle'))
    deltaThetaAngle = property(lambda self:self.att('deltaThetaAngle'))

    def __repr__(self):
        linked = getattr(self,'PosXYZ', None)
        return "sphere %20s : %s :  %s " % (self.name, self.outerRadius, self.transform)

    def asrev(self):
        xyz = self.xyz 
        ro = self.outerRadius.value
        ri = self.innerRadius.value
        st = self.startThetaAngle.value
        dt = self.deltaThetaAngle.value
        sz = None
        wi = None
        if ri is not None and ri > 0:
            wi = ro - ri 

        return [Rev('Sphere', self.name,xyz, ro, sz, st, dt, wi)]

class Tubs(Primitive):
    sizeZ = property(lambda self:self.att('sizeZ'))

    def __repr__(self):
        return "Tubs %20s : outerRadius %s  sizeZ %s  : transform  %s " % (self.name, self.outerRadius, self.sizeZ, self.transform)

    def asrev(self):
        sz = self.sizeZ.value
        r = self.outerRadius.value
        xyz = self.xyz 
        return [Rev('Tubs', self.name,xyz, r, sz )]


class PosXYZ(Elem):
    x = property(lambda self:self.att('x',0))
    y = property(lambda self:self.att('y',0))
    z = property(lambda self:self.att('z',0))
    def __repr__(self):
        return "PosXYZ  %s  " % (repr(self.z))



class Context(object):
    def __init__(self, d, expand):
        self.d = d
        self.expand = expand

    def build_context(self, params):
        name_error = params
        for wave in range(3):
            name_error, type_error = self._build_context(name_error, wave)
            log.info("after wave %s remaining name_error %s type_error %s " % (wave, len(name_error), len(type_error)))

    def evaluate(self, expr):
        txt = "float(%s)" % expr
        try:
            val = eval(txt, globals(), self.d)
        except NameError, ex:
            log.fatal("%s :failed to evaluate expr %s " % (repr(ex), expr))
            val = None 
 
        return val    

    def _build_context(self, params, wave):
        name_error = []
        type_error = []
        for p in params:
            if p.expr in self.expand:
                expr = self.expand[p.expr]
                log.warn("using manual expansion of %s to %s " % (p.expr, expr))
            else:
                expr = p.expr

            txt = "float(%s)" % expr
            try:
                val = eval(txt, globals(), self.d)
                pass
                self.d[p.name] = float(val)  
            except NameError:
                name_error.append(p)
                log.info("NameError %s %s " % (p.name, txt ))
            except TypeError:
                type_error.append(p)
                log.info("TypeError %s %s " % (p.name, txt ))
            pass
        return name_error, type_error
          
    def dump_context(self, prefix):
        log.info("dump_context %s* " % prefix ) 
        return "\n".join(["%25s : %s " % (k,v) for k,v in filter(lambda kv:kv[0].startswith(prefix),self.d.items())])
 
    def __repr__(self):
        return "\n".join(["%25s : %s " % (k,v) for k,v in self.d.items()])



class Dddb(Elem):
    kls = {
        "parameter":Parameter,
        "sphere":Sphere,
        "tubs":Tubs,
        "logvol":Logvol,
        "physvol":Physvol,
        "posXYZ":PosXYZ,
        "intersection":Intersection,
        "union":Union,
    }

    primitive = [Sphere, Tubs]
    composite = [Union, Intersection, Physvol]
    transform = [PosXYZ]

    expand = {
        "(PmtHemiFaceROCvac^2-PmtHemiBellyROCvac^2-(PmtHemiFaceOff-PmtHemiBellyOff)^2)/(2*(PmtHemiFaceOff-PmtHemiBellyOff))":
         "(PmtHemiFaceROCvac*PmtHemiFaceROCvac-PmtHemiBellyROCvac*PmtHemiBellyROCvac-(PmtHemiFaceOff-PmtHemiBellyOff)*(PmtHemiFaceOff-PmtHemiBellyOff))/(2*(PmtHemiFaceOff-PmtHemiBellyOff))" 

    }


    @classmethod
    def parse(cls, path):
        g = Dddb(parse_(path))
        g.init()
        return g

    def __call__(self, expr):
        return self.ctx.evaluate(expr)

    def init(self):
        self.g = self

        pctx = {}
        pctx["mm"] = 1.0 
        pctx["cm"] = 10.0 
        pctx["m"] = 1000.0 
        pctx["degree"] = 1.0
        pctx["radian"] = 180./math.pi

        self.ctx = Context(pctx, self.expand)
        self.ctx.build_context(self.params_())
        self.ctx.dump_context('PmtHemi')

    def logvol_(self, name):
        return self.find_(".//logvol[@name='%s']"%name)

    def logvols_(self):
        return self.findall_(".//logvol")

    def params_(self, prefix=None):
        pp = self.findall_(".//parameter")
        if prefix is not None:
            pp = filter(lambda p:p.hasprefix(prefix), pp)
        return pp  

    def context_(self, prefix=None):
        dd = {}
        for k,v in self.ctx.d.items():
            if k.startswith(prefix) or prefix is None:
                dd[k] = v
        return dd   

    def dump_context(self, prefix=None):
        print pp_(self.context_(prefix))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    g.dump_context('PmtHemi')

    lv = g.logvol_("lvPmtHemi")

    for maxdepth in range(4):
        revs = lv.allrev(maxdepth=maxdepth) 
        log.info("maxdepth %s returned %s revs " % (maxdepth, len(revs)))
        print "\n".join(map(str,revs))



