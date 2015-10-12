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
    name  = property(lambda self:self.elem.attrib['name'])

    @classmethod
    def link_PosXYZ(cls, ls, geom ):
        """
        # attach any PosXYZ instances in the list to preceeding geometry elements
        """
        for i in range(len(ls)):
            if type(ls[i]) == PosXYZ and type(ls[i-1]) in cls.geom:
                setattr(ls[i-1],'PosXYZ', ls[i]) 
                log.debug("linking %s to %s " % (ls[i], ls[i-1]))


    def __init__(self, elem, g=None):
        self.elem = elem 
        self.g = g 

    def att(self, k, evaluate=True):
        v = self.elem.attrib.get(k, None)
        return Att(v, self.g, evaluate=evaluate) if v is not None else None 

    def findall_(self, expr):
        return map( lambda e:self.g.kls.get(e.tag,Elem)(e,self.g), self.elem.findall(expr) )

    def findone_(self, expr):
        all_ = self.findall_(expr)
        assert len(all_) == 1
        return all_[0]

    def find_(self, expr):
        e = self.elem.find(expr) 
        return self.g.kls.get(e.tag,Elem)(e,self.g) if e is not None else None

    def __repr__(self):
        return "%15s : %s " % ( self.elem.tag, repr(self.elem.attrib) )

    def asrev(self):
        w = self.outerRadius.value
        x,y,z = 0,0,0
        if hasattr(self, 'PosXYZ'):
            z = self.PosXYZ.z.value
        return [x,y,z,w], self.__class__.__name__

    def allrev(self):
        """
        Assuming fairly simple
        """
        comps = self.findall_(".//*") 
        self.link_PosXYZ(comps)

        shapes = filter(lambda _:type(_) in self.geom, comps)
        other = filter(lambda _:type(_) not in self.geom + [PosXYZ], comps)
        assert len(other) == 0, other 

        shs = []
        for sh in shapes:
           shs.append(sh.asrev())
        pass
        return shs




class Parameter(Elem):
    expr = property(lambda self:self.elem.attrib['value'])

    def hasprefix(self, prefix):
        return self.name.startswith(prefix)

    def __repr__(self):
        return "%30s : %s " % ( self.name, self.expr )

class Logvol(Elem):
    def __repr__(self):
        a = self.elem.attrib
        return "%30s %20s %s " % (a['name'], a.get('material',"-"), a.get("sensdet","-"))

    def union(self):
        return self.findone_(".//union")

    def intersection(self):
        return self.findone_(".//intersection")


class Union(Elem):
    def __repr__(self):
        return "Union %20s  " % (self.name)


class Intersection(Elem):
    def __repr__(self):
        return "Intersection %20s  " % (self.name)


 
class Sphere(Elem):
    outerRadius = property(lambda self:self.att('outerRadius'))
    def __repr__(self):
        linked = getattr(self,'PosXYZ', None)
        return "sphere %20s : %s :  %s " % (self.name, self.outerRadius, linked)

class Tubs(Elem):
    outerRadius = property(lambda self:self.att('outerRadius'))
    innerRadius = property(lambda self:self.att('innerRadius'))
    sizeZ = property(lambda self:self.att('sizeZ'))
    def __repr__(self):
        linked = getattr(self,'PosXYZ', None)
        return "Tubs %20s : outerRadius %s  sizeZ %s  : linked  %s " % (self.name, self.outerRadius, self.sizeZ, linked)

class PosXYZ(Elem):
    z = property(lambda self:self.att('z'))
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
                #if p.name.startswith('Pmt'):
                #    log.info(" %s : %s => %s " % (p.name, txt, val ))
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
        return "\n".join(["%25s : %s " % (k,v) for k,v in filter(lambda kv:kv[0].startswith(prefix),self.d.items())])
 
    def __repr__(self):
        return "\n".join(["%25s : %s " % (k,v) for k,v in self.d.items()])



class Dddb(Elem):
    kls = {
        "parameter":Parameter,
        "sphere":Sphere,
        "logvol":Logvol,
        "posXYZ":PosXYZ,
        "intersection":Intersection,
        "union":Union,
    }

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

    hemi = g.logvol_("lvPmtHemi")
    inter = hemi.intersection()
    revs = inter.allrev() 

    print revs

