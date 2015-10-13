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
   if d == 0:
       return None

   x = (d*d - r*r + R*R)/(2.*d)
   return x + xy_a[0] 



def plot_revs(ax, revs):
    edgecolor = ['r','g','b']

    zmin = 1e6
    zmax = -1e6
  
    ymin = 1e6
    ymax = -1e6

    circles = []
    rects = []

    for i,rev in enumerate(revs):

        xyz,r,sz = rev.xyz, rev.r, rev.sz
        z = xyz[2]

        if rev.typ == "Sphere":
            circ = plt.Circle((z,0.),r, edgecolor=edgecolor[i%len(edgecolor)], facecolor='w', alpha=0.5)
            circles.append(circ)

            if z-r < zmin:
                zmin = z-r
            if z+r > zmax:
                zmax = z+r
            if -r < ymin:
                ymin = -r
            if r > ymax:
                ymax = r

        elif rev.typ == "Tubs":
             botleft = (z,-r)
             width = sz
             height = 2*r
             log.info("rect %s %s %s " % (botleft, width, height ))
             rect = plt.Rectangle(botleft, width, height, edgecolor='r', facecolor='w', alpha=0.5) 
             rects.append(rect)   
        pass
    pass

    lx = []
    lz = []
    for i in range(1,len(circles)):
        ci = circle_intersect(circles[i-1],circles[i])
        if ci is not None:
            lx.append([ci,ci])      # x,z coords are start/end of line segment
            lz.append([zmin,zmax])

    ax.set_xlim(zmin,zmax)
    ax.set_ylim(ymin,ymax)

    for sh in circles + rects:
        ax.add_artist(sh)

    for i in range(len(lx)):
        a = 0.1 if i == 1 else 0.9
        log.info(" i %s : lx %s lz %s " % (i, repr(lx[i]), repr(lz[i]))) 
        line = lines.Line2D( lx[i], lz[i] , lw=5., alpha=a)
        ax.add_line(line)

    #ax.autoscale_view(True,True,True)




if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    g.dump_context('PmtHemi')

    revs = []

    # lvPmtHemiCathode is 2 spheres differing in radius by 0.05 mm !
    #        PmtHemiCathodeThickness : 0.050000 
    #
    #lvns = "lvPmtHemi lvPmtHemiVacuum lvPmtHemiCathode"
    lvns = "lvPmtHemiCathode"

    for lvn in lvns.split():
        lv = g.logvol_(lvn)
        union = lv.union()  # hmm get this from logvol without knowning content 
        revs += union.allrev()
    pass

    print "\n".join(map(str,revs))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plot_revs(ax,revs)
    fig.show()




