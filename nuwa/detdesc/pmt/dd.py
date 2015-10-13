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
    name  = property(lambda self:self.elem.attrib['name'])
    is_primitive = property(lambda self:type(self) in self.g.primitive)
    is_composite = property(lambda self:type(self) in self.g.composite)
    is_transform = property(lambda self:type(self) in self.g.transform)


    @classmethod
    def link_transform(cls, ls):
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

    def allrev(self):
        """
        Assuming fairly simple
        """
        components = self.findall_(".//*") 
        self.link_transform(components)

        revs = []
        for c in components:
            if c.is_primitive:
                xrev = c.asrev()
                log.info("allrev: primitive : %s " % repr(xrev)) 
                revs.extend(xrev)
            elif c.is_composite:
                xrev = c.allrev() 
                log.info("allrev: composite : %s " % repr(xrev)) 
                revs.extend(xrev)
            elif c.is_transform:
                pass
            else:
                log.warning("skipped component %s " % repr(c))
            pass
                  
        return revs



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




class Rev(object):
    def __init__(self, typ, xyz, r, sz=None):
        self.typ = typ
        self.xyz = xyz 
        self.r = r
        self.sz = sz

    def __repr__(self):
        return "Rev('%s', %s, %s, %s)" % (self.typ, self.xyz, self.r, self.sz)



class Primitive(Elem):
    is_rev = True
    outerRadius = property(lambda self:self.att('outerRadius'))
    innerRadius = property(lambda self:self.att('innerRadius'))


class Sphere(Primitive):
    """
    What convention for theta,phi ? 

    http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/Detector/geomSolids.html

    """
    startThetaAngle = property(lambda self:self.att('startThetaAngle'))
    deltaThetaAngle = property(lambda self:self.att('deltaThetaAngle'))

    def __repr__(self):
        linked = getattr(self,'PosXYZ', None)
        return "sphere %20s : %s :  %s " % (self.name, self.outerRadius, self.transform)

    def asrev(self):

        xyz = self.xyz 
        ro = self.outerRadius.value
        ri = self.innerRadius.value

        xrev = []
        xrev += [Rev('Sphere', xyz, ro )]
        if ri is not None and ri > 0:
            xrev += [Rev('Sphere', xyz, ri )]
        
        return xrev




class Tubs(Elem):
    is_rev = True
    outerRadius = property(lambda self:self.att('outerRadius'))
    innerRadius = property(lambda self:self.att('innerRadius'))
    sizeZ = property(lambda self:self.att('sizeZ'))

    def __repr__(self):
        return "Tubs %20s : outerRadius %s  sizeZ %s  : transform  %s " % (self.name, self.outerRadius, self.sizeZ, self.transform)

    def asrev(self):
        sz = self.sizeZ.value
        r = self.outerRadius.value
        xyz = self.xyz 
        return [Rev('Tubs', xyz, r, sz )]





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
        "tubs":Tubs,
        "logvol":Logvol,
        "posXYZ":PosXYZ,
        "intersection":Intersection,
        "union":Union,
    }

    primitive = [Sphere, Tubs]
    composite = [Union, Intersection]
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

    hemi = g.logvol_("lvPmtHemi")
    inter = hemi.intersection()
    revs = inter.allrev() 

    print revs

