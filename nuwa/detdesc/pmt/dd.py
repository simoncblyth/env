#!/usr/bin/env python
import os, re, logging, math
import numpy as np
import lxml.etree as ET
import lxml.html as HT
from math import acos

log = logging.getLogger(__name__)

tostring_ = lambda _:ET.tostring(_)
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])

X,Y,Z = 0,1,2

class Att(object):
    def __init__(self, expr, g, evaluate=True):
        self.expr = expr
        self.g = g  
        self.evaluate = evaluate

    value = property(lambda self:self.g.ctx.evaluate(self.expr) if self.evaluate else self.expr)

    def __repr__(self):
        return "%s : %s " % (self.expr, self.value)

class Elem(object):
    posXYZ = None
    is_rev = False
    name  = property(lambda self:self.elem.attrib.get('name',None))

    # structure avoids having to forward declare classes
    is_primitive = property(lambda self:type(self) in self.g.primitive)
    is_composite = property(lambda self:type(self) in self.g.composite)
    is_intersection = property(lambda self:type(self) in self.g.intersection)
    is_tubs = property(lambda self:type(self) in self.g.tubs)
    is_sphere = property(lambda self:type(self) in self.g.sphere)
    is_union = property(lambda self:type(self) in self.g.union)
    is_posXYZ = property(lambda self:type(self) in self.g.posXYZ)
    is_geometry  = property(lambda self:type(self) in self.g.geometry)

    @classmethod
    def link_posXYZ(cls, ls, posXYZ):
        """
        Attach *posXYZ* attribute to all primitives in the list 
        """
        for i in range(len(ls)):
            if ls[i].is_primitive:
                ls[i].posXYZ = posXYZ 
                log.debug("linking %s to %s " % (posXYZ, ls[i]))

    @classmethod
    def link_prior_posXYZ(cls, ls):
        """
        Attach any *posXYZ* instances in the list to preceeding primitives
        """
        for i in range(len(ls)):
            if ls[i].is_posXYZ and ls[i-1].is_primitive:
                ls[i-1].posXYZ = ls[i] 
                log.debug("linking %s to %s " % (ls[i], ls[i-1]))

    def _get_desc(self):
        return "%10s %15s %s " % (type(self).__name__, self.xyz, self.name )
    desc = property(_get_desc)

    def _get_xyz(self):
       x = y = z = 0
       if self.posXYZ is not None:
           x = self.posXYZ.x.value 
           y = self.posXYZ.y.value 
           z = self.posXYZ.z.value 
       pass
       return [x,y,z] 
    xyz = property(_get_xyz)

    def _get_z(self):
       """
       z value from any linked *posXYZ* or 0 
       """
       z = 0
       if self.posXYZ is not None:
           z = self.posXYZ.z.value 
       return z
    z = property(_get_z)

    def __init__(self, elem, g=None):
        self.elem = elem 
        self.g = g 

    def att(self, k, dflt=None):
        v = self.elem.attrib.get(k, None)
        return Att(v, self.g) if v is not None else Att(dflt, self.g, evaluate=False) 

    def findall_(self, expr):
        """
        lxml findall result elements are wrapped in the class appropriate to their tags 
        """
        wrap_ = lambda e:self.g.kls.get(e.tag,Elem)(e,self.g)
        fa = map(wrap_, self.elem.findall(expr) )
        kln = self.__class__.__name__
        name = self.name 
        log.debug("findall_ from %s:%s expr:%s returned %s " % (kln, name, expr, len(fa)))
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

    def children(self):
        """
        Defines the nature of the tree. 

        * for Physvol returns single item list containing the referenced Logvol
        * for Logvol returns list of all contained Physvol
        * otherwise returns empty list 

        NB bits of geometry of a Logvol are not regarded as children, 
        but rather are constitutent to it.
        """
        if type(self) is Physvol:
            lvn = self.logvolref.split("/")[-1]
            lv = self.g.logvol_(lvn)
            return [lv]
        elif type(self) is Logvol:
            pvs = self.findall_("./physvol")
            return pvs
        else:
            return []  

    def partition_intersection_3spheres(self, spheres):
        """
        :param spheres:  list of three *Sphere* in ascending center z order, which are assumed to intersect
        :return parts:  list of three sphere *Part* instances 

        Extend the below two sphere intersection approach to three spheres
        numbered s1,s2,s3 from left to right, with two ZPlane intersections z23, z12.

        left
            from s3, bounded by z23 on right  

        middle
            from s2, bounded by z23 on left, z12 on right

        right 
            from s1, bounded by z12 on left

        """
        s1, s2, s3 = spheres

        assert s1.z < s2.z < s3.z

        z12 = Sphere.intersect("z12",s1,s2)   # ZPlane of s1 s2 intersection
        z23 = Sphere.intersect("z23",s2,s3)   # ZPlane of s2 s3 intersection

        assert z23.z < z12.z 

        p1 = s3.part_zleft(z23)       
        p2 = s2.part_zmiddle(z23, z12)       
        p3 = s1.part_zright(z12)       

        assert p1.bbox.z < p2.bbox.z < p3.bbox.z

        return [p3,p2,p1]

    def partition_intersection_2spheres(self, spheres):
        """
        :param spheres: list of two *Sphere* in ascending center z order
        :return parts: list of two sphere *Part* instances

        Consider splitting the lens shape made from the intersection of two spheres
        along the plane of the intersection. 
        The left part of the lens comes from the right Sphere 
        and the right part comes left Sphere.
        """ 
        s1, s2 = spheres

        assert s1.z < s2.z 

        z12 = Sphere.intersect("z12",s1,s2) 
        p1 = s2.part_zleft(z12)       
        p2 = s1.part_zright(z12)       

        assert p1.bbox.z < p2.bbox.z

        return [p2,p1]

    def partition_intersection(self):
        log.info(self)

        spheres = []
        comps = self.findall_("./*")
        self.link_prior_posXYZ(comps)

        other = []
        for c in comps:
            if type(c) is Sphere:
                spheres.append(c)
            elif type(c) is PosXYZ:
                pass
            else:
                other.append(c)
            pass

        assert len(other) == 0, "only 2/3-sphere intersections handled"    

        for i,s in enumerate(spheres):
            log.debug("s%d: %s %s " % (i, s.desc, s.outerRadius.value))
 
        if len(spheres) == 3:
            return self.partition_intersection_3spheres(spheres) 
        elif len(spheres) == 2:
            return self.partition_intersection_2spheres(spheres) 
        else:
            assert 0 


    def partition_union(self):
        """
        union of a 3-sphere lens shape and a tubs requires:

        * adjust bbox of the abutting part Sphere to the intersection z of tubs and Sphere
        * avoid a surface at the interface of tubs endcap and part Sphere

        """
        log.info(self)
        comps = self.findall_("./*")
        self.link_prior_posXYZ(comps)

        rparts = []
        if len(comps) == 3 and comps[0].is_intersection and comps[1].is_tubs and comps[2].is_posXYZ:
        
            sparts = Part.ascending_bbox_zleft(comps[0].partition_intersection())
            tpart = comps[1].as_part()
            ts = Part.intersect_tubs_sphere("ts", tpart, sparts[0], -1)   # -ve root for leftmost
            log.info("ts %s " % repr(ts))

            for i, s in enumerate(sparts):
                log.info("sp(%s) %s " % (i, repr(s)))

            log.info("tp(0) %s " % repr(tpart))

            sparts[0].bbox.zleft = ts.z   
            tpart.bbox.zright = ts.z
            tpart.enable_endcap("P")  # smaller Z endcap 

            rparts.extend(sparts)
            rparts.extend([tpart])
        else:
            xret = self.parts()   
            rparts.extend(xret)
        pass
        return rparts  ; 


    def parts(self):
        """
        Provides parts from a single LV only, ie not
        following pv refs. Recursion is needed 
        in order to do link posXYZ transforms with geometry
        and skip them from the parts returned.
        """
        if type(self) is Physvol:
            return [] 

        comps = self.findall_("./*")  # one lev only
        self.link_prior_posXYZ(comps)

        rparts = []
        for c in comps:
            if c.is_primitive:
                rparts.extend([c.as_part()])  # assume in union, so no need for chopping ?
            elif c.is_intersection:
                xret = c.partition_intersection() 
                rparts.extend(xret)
            elif c.is_union:
                xret = c.partition_union() 
                rparts.extend(xret)
            elif c.is_composite:
                xret = c.parts() 
                rparts.extend(xret)
            elif c.is_posXYZ:
                pass
            else:
                log.warning("skipped component %s " % repr(c))
            pass

        return rparts

    def geometry(self):
        return filter(lambda c:c.is_geometry, self.components())

    def allrev(self, depth=0, maxdepth=10):
        assert 0, "no longer using this approach as not enough control over the recursion"
 
        if depth > maxdepth:
            return []

        if type(self) is Physvol:
            lvn = self.logvolref.split("/")[-1]
            log.info("physvol is special logvolref %s lvn %s " % (self.logvolref, lvn ))
            lv = self.g.logvol_(lvn)
            depth += 1
            components = lv.findall_("./*")  
            posXYZ = self.find_("./posXYZ") 
            if posXYZ:
                self.link_posXYZ(components, posXYZ)
        else:
            components = self.findall_("./*")  # one lev only
            self.link_prior_posXYZ(components)
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
            elif c.is_posXYZ:
                pass
            else:
                log.warning("skipped component %s " % repr(c))
            pass
                  
        return revs
    
 
class Logvol(Elem):
    material = property(lambda self:self.elem.attrib.get('material', None))
    sensdet = property(lambda self:self.elem.attrib.get('sensdet', None))
    def __repr__(self):
        return "LV %-20s %20s %s " % (self.name, self.material, self.sensdet)

class Physvol(Elem):
    logvolref = property(lambda self:self.elem.attrib.get('logvol', None))
    def __repr__(self):
        return "PV %-20s %s " % (self.name, self.logvolref)



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


class ZPlane(object):
    def __init__(self, name, z, y):
        self.name = name
        self.z = z
        self.y = y 
    def __repr__(self):
        return "ZPlane %s z:%s y:%s " % (self.name, self.z, self.y )

class Part(object):
    @classmethod 
    def ascending_bbox_zleft(cls, parts):
        return sorted(parts, key=lambda p:p.bbox.zleft)

    @classmethod 
    def intersect_tubs_sphere(cls, name, tubs, sphere, sign ):
        """
        :param name: identifier of ZPlane created
        :param tubs: tubs Part instance  
        :param sphere: sphere Part instance  
        :param sign: 1 or -1 sign of the sqrt  

        Sphere at zp on Z axis

            xx + yy + (z-zp)(z-zp) = RR    

        Cylinder along Z axis from -sizeZ/2 to sizeZ/2
           
            xx + yy = rr

        Intersection is a circle in Z plane  

            (z-zp) = sqrt(RR - rr) 

        """ 
        R = sphere.radius 
        r = tubs.radius

        RR_m_rr = R*R - r*r
        assert RR_m_rr > 0

        iz = math.sqrt(RR_m_rr)  
        assert iz < tubs.sizeZ 

        return ZPlane(name, sphere.xyz[Z] + sign*iz, r) 

    def enable_endcap(self, tag):
        ENDCAP_P = 0x1 <<  0
        ENDCAP_Q = 0x1 <<  1 
        pass
        if tag == "P":
            self.flags |= ENDCAP_P 
        elif tag == "Q":
            self.flags |= ENDCAP_Q 
        else:
            log.warning("tag is not P or Q, for the low Z endcap (P) and higher Z endcap (Q)")


    def __init__(self, typ, name, xyz, radius, sizeZ=0.):
        """
        see cu/hemi-pmt.cu for where these are used 
        """
        self.typ = typ
        self.name = name
        self.xyz = xyz
        self.radius = radius
        self.sizeZ = sizeZ   # used for Tubs
        self.bbox = None

        self.flags = 0
        # Tubs endcap control

        if typ == 'Sphere':
            self.typecode = 1
        elif typ == 'Tubs':
            self.typecode = 2
        else:
            assert 0

    def __repr__(self):
        return "Part %s %s %s r:%s sz:%s bb:%s" % (self.typ, self.name, repr(self.xyz), self.radius, self.sizeZ, repr(self.bbox)) 

    def as_quads(self, bbscale=1):
        quads = []
        quads.append( [self.xyz[0], self.xyz[1], self.xyz[2], self.radius] )
        quads.append( [self.sizeZ, 0, 0, 0] )
        for q in self.bbox.as_quads(scale=bbscale):
            quads.append(q)
        return quads
           

class BBox(object):
    def __init__(self, min_, max_):
        self.min_ = np.array(min_)
        self.max_ = np.array(max_)

    def _get_zleft(self):
        return self.min_[Z]
    def _set_zleft(self, val):
        self.min_[Z] = val 
    zleft = property(_get_zleft, _set_zleft)

    def _get_zright(self):
        return self.max_[Z]
    def _set_zright(self, val):
        self.max_[Z] = val 
    zright = property(_get_zright, _set_zright)

 
    x = property(lambda self:(self.min_[X] + self.max_[X])/2.)
    y = property(lambda self:(self.min_[Y] + self.max_[Y])/2.)
    z = property(lambda self:(self.min_[Z] + self.max_[Z])/2.)
    xyz = property(lambda self:[self.x, self.y,self.z])

    def as_quads(self,scale=1):
        qmin = np.zeros(4)
        qmin[:3] = self.min_*scale
        qmax = np.zeros(4)
        qmax[:3] = self.max_*scale
        return qmin, qmax 

    def __repr__(self):
        return "BBox min:%s max:%s xyz:%s" % (repr(self.min_), repr(self.max_), repr(self.xyz) )




class Primitive(Elem):
    is_rev = True
    outerRadius = property(lambda self:self.att('outerRadius'))
    innerRadius = property(lambda self:self.att('innerRadius'))

    def bbox(self, zl, zr, yn, yp ):
        assert yn < 0 and yp > 0 and zr > zl
        return BBox([yn,yn,zl], [yp,yp,zr])
 

class Sphere(Primitive):
    startThetaAngle = property(lambda self:self.att('startThetaAngle'))
    deltaThetaAngle = property(lambda self:self.att('deltaThetaAngle'))

    @classmethod
    def intersect(cls, name, a_, b_ ):
        """
        Find Z intersect of two Z offset spheres 

        * http://mathworld.wolfram.com/Circle-CircleIntersection.html

        :param name: identifer passed to ZPlane
        :param a_: *Sphere* instance
        :param b_: *Sphere* instance

        :return zpl: *ZPlane* instance with z and y attributes where:
                     z is intersection coordinate 
                     y is radius of the intersection circle 
        """
        R = a_.outerRadius.value
        r = b_.outerRadius.value
        a = a_.xyz
        b = b_.xyz

        log.debug(" R %s a %s " % ( R, repr(a)) )  
        log.debug(" r %s b %s " % ( r, repr(b)) )  

        dx = b[X] - a[X]
        dy = b[Y] - a[Y]
        dz = b[Z] - a[Z]

        assert dx == 0
        assert dy == 0
        assert dz != 0 

        d = dz             # use Sphere a_ frame
        dd_m_rr_p_RR = d*d - r*r + R*R 
        z = dd_m_rr_p_RR/(2.*d)
        yy = (4.*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.*d*d)
        y = math.sqrt(yy)

        # add a[Z] to return to original frame
        return ZPlane(name, z+a[Z], y ) 

    def as_part(self):
        radius = self.outerRadius.value 
        z = self.xyz[2]
        p = Part('Sphere', self.name + "_part", self.xyz, radius )
        p.bbox = self.bbox(z-radius, z+radius, -radius, radius)
        return p 

    def part_zleft(self, zpl):
        radius = self.outerRadius.value 
        z = self.xyz[2]
        ymax = zpl.y 
        p = Part('Sphere', self.name + "_part_zleft", self.xyz, radius )
        p.bbox = self.bbox(z-radius, zpl.z, -ymax, ymax)
        return p 

    def part_zright(self, zpr):
        radius = self.outerRadius.value 
        z = self.xyz[2]
        ymax = zpr.y 
        p = Part('Sphere', self.name + "_part_zright", self.xyz, radius )
        p.bbox = self.bbox(zpr.z,z+radius, -ymax, ymax)
        return p 

    def part_zmiddle(self, zpl, zpr):
        p = Part('Sphere', self.name + "_part_zmiddle", self.xyz, self.outerRadius.value )
        ymax = max(zpl.y,zpr.y)
        p.bbox = self.bbox(zpl.z,zpr.z,-ymax,ymax )
        return p 

    def __repr__(self):
        return "sphere %20s : %s :  %s " % (self.name, self.outerRadius, self.posXYZ)

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
        return "Tubs %20s : outerRadius %s  sizeZ %s  :  %s " % (self.name, self.outerRadius, self.sizeZ, self.posXYZ)

    def as_part(self):
        sizeZ = self.sizeZ.value
        radius = self.outerRadius.value 
        z = self.xyz[2]
        p = Part('Tubs', self.name + "_part", self.xyz, radius, sizeZ )
        p.bbox = self.bbox(z-sizeZ/2, z+sizeZ/2, -radius, radius)
        return p 

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
        type_error = []
        for wave in range(3):
            name_error, type_error = self._build_context(name_error, wave)
            log.debug("after wave %s remaining name_error %s type_error %s " % (wave, len(name_error), len(type_error)))
        pass
        assert len(name_error) == 0
        assert len(type_error) == 0

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
                log.debug("using manual expansion of %s to %s " % (p.expr, expr))
            else:
                expr = p.expr

            txt = "float(%s)" % expr
            try:
                val = eval(txt, globals(), self.d)
                pass
                self.d[p.name] = float(val)  
            except NameError:
                name_error.append(p)
                log.debug("NameError %s %s " % (p.name, txt ))
            except TypeError:
                type_error.append(p)
                log.debug("TypeError %s %s " % (p.name, txt ))
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
    intersection = [Intersection]
    union = [Union]
    tubs = [Tubs]
    sphere = [Sphere]
    composite = [Union, Intersection, Physvol]
    geometry = [Sphere, Tubs, Union, Intersection]
    posXYZ= [PosXYZ]

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

    def logvol_(self, name):
        if name[0] == '/':
            name = name.split("/")[-1]
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
    format_ = "[%(filename)s +%(lineno)3s %(funcName)20s() ] %(message)s" 
    logging.basicConfig(level=logging.INFO, format=format_)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    g.dump_context('PmtHemi')

    lv = g.logvol_("lvPmtHemi")

    for maxdepth in range(4):
        revs = lv.allrev(maxdepth=maxdepth) 
        log.info("maxdepth %s returned %s revs " % (maxdepth, len(revs)))
        print "\n".join(map(str,revs))



