#!/usr/bin/env python
"""
::


    In [5]: g.logvols_()
    Out[5]: 
    [                lvPmtHemiFrame                    - - ,
                          lvPmtHemi                Pyrex - ,
                lvPmtHemiwPmtHolder                    - - ,
                      lvAdPmtCollar   UnstStainlessSteel - ,
                    lvPmtHemiVacuum               Vacuum - ,
                   lvPmtHemiCathode             Bialkali DsPmtSensDet ,
                    lvPmtHemiBottom         OpaqueVacuum - ,
                    lvPmtHemiDynode         OpaqueVacuum - ,
                           lvPmtTee   UnstStainlessSteel - ,
                       lvPmtTopRing   UnstStainlessSteel - ,
                      lvPmtBaseRing   UnstStainlessSteel - ,
                        lvMountRib1   UnstStainlessSteel - ,
                        lvMountRib2   UnstStainlessSteel - ,
                        lvMountRib3   UnstStainlessSteel - ,
                    lvMountRib1unit                    - - ,
                    lvMountRib2unit                    - - ,
                    lvMountRib3unit                    - - ,
                       lvMountRib1s                    - - ,
                       lvMountRib2s                    - - ,
                       lvMountRib3s                    - - ,
                         lvPmtMount                    - - ]

    In [6]: hemiVac = g.logvol_("lvPmtHemiVacuum")

    In [7]: hemiVac
    Out[7]:                lvPmtHemiVacuum               Vacuum - 

    In [8]: hemiVac.findall_("*")
    Out[8]: 
    [          union : {'name': 'pmt-hemi-vac'} ,
             physvol : {'logvol': '/dd/Geometry/PMT/lvPmtHemiCathode', 'name': 'pvPmtHemiCathode'} ,
             physvol : {'logvol': '/dd/Geometry/PMT/lvPmtHemiBottom', 'name': 'pvPmtHemiBottom'} ,
             physvol : {'logvol': '/dd/Geometry/PMT/lvPmtHemiDynode', 'name': 'pvPmtHemiDynode'} ]





"""
from dd import *
import matplotlib.pyplot as plt 
import matplotlib.lines as lines
import matplotlib.patches as patches

log = logging.getLogger(__name__)



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



def plot_revolution(ax, shapes):
    edgecolor = ['r','g','b']
    spheres = filter(lambda _:_[-1] > 0, shapes) 

    zmin = 1e6
    zmax = -1e6
  
    ymin = 1e6
    ymax = -1e6

    circles = []
    for i,s in enumerate(spheres):
        c = plt.Circle((s[2],0.),s[3], edgecolor=edgecolor[i%len(edgecolor)], facecolor='w', alpha=0.5)
        circles.append(c)

        if s[2]-s[3] < zmin:
            zmin = s[2]-s[3]
        if s[2]+s[3] > zmax:
            zmax = s[2]+s[3]
        if -s[3] < ymin:
            ymin = -s[3]
        if s[3] > ymax:
            ymax = s[3]

        pass
    pass

    lx = []
    lz = []
    for i in range(1,len(circles)):
        ci = circle_intersect(circles[i-1],circles[i])
        lx.append([ci,ci])      # x,z coords are start/end of line segment
        lz.append([zmin,zmax])

    ax.set_xlim(zmin,zmax)
    ax.set_ylim(ymin,ymax)

    for c in circles:
        ax.add_artist(c)

    for i in range(len(lx)):
        a = 0.1 if i == 1 else 0.9
        log.info(" i %s : lx %s lz %s " % (i, repr(lx[i]), repr(lz[i]))) 
        line = lines.Line2D( lx[i], lz[i] , lw=5., alpha=a)
        ax.add_line(line)

    #ax.autoscale_view(True,True,True)




def pmt_hemi_pyrex(g):

    # spheres  

    xy1 = (0.,0.)
    xy2 = (g('PmtHemiFaceOff-PmtHemiBellyOff'),0.)
    xy3 = (g('PmtHemiFaceOff+PmtHemiBellyOff'),0.)

    r1 = g('PmtHemiFaceROC')
    r2 = g('PmtHemiBellyROC')
    r3 = g('PmtHemiBellyROC')

    # tubs
  
    bxy = [g('-0.5*PmtHemiGlassBaseLength'),-1.*g('PmtHemiGlassBaseRadius')]
    bwh = [g('PmtHemiGlassBaseLength'), 2.0*g('PmtHemiGlassBaseRadius')]   
 
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
        line = lines.Line2D( l1[i], l2[i] , lw=5., alpha=a)
        ax.add_line(line)

    ax.autoscale_view(True,True,True)
    fig.show()

    fig.savefig('/tmp/pmt_hemi_pyrex.png')


def pyrex(g):
    hemi = g.logvol_('lvPmtHemi')
    for _ in hemi.findall_("./union/*"):
        print _.__class__.__name__, _
    pmt_hemi_pyrex(g)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    g.dump_context('PmtHemi')

    hemiVac = g.logvol_("lvPmtHemiVacuum")

    lv = g.logvol_("lvPmtHemi")

    # need generic way to pull the shapes 
    uni = lv.union()

    itr = lv.intersection()
    sphs = itr.allspheres() 
    print sphs

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plot_revolution(ax,sphs)
    fig.show()




