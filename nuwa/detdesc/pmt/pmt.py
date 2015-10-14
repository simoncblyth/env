#!/usr/bin/env python
"""
::


"""
from dd import *
import math
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

log = logging.getLogger(__name__)



class Circ(object):
    @classmethod
    def intersect(cls, a, b ):
        """
        http://mathworld.wolfram.com/Circle-CircleIntersection.html
        """
        R = a.radius
        r = b.radius

        xy_a = a.pos
        xy_b = b.pos

        log.debug(" A %s xy %s " % ( R, repr(xy_a)) )  
        log.debug(" B %s xy %s " % ( r, repr(xy_b)) )  

        assert(xy_b[1] == xy_a[1]) 
        d = xy_b[0] - xy_a[0]
        if d == 0:
            return None

        dd_m_rr_p_RR = d*d - r*r + R*R 

        x = dd_m_rr_p_RR/(2.*d)
        yy = (4.*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.*d*d)
        y = math.sqrt(yy)

        npos = [x + xy_a[0], -y]
        ppos = [x + xy_a[0],  y]

        return Chord(npos, ppos, a, b ) 

    def __init__(self, pos, radius, startTheta=None, deltaTheta=None, width=None):
        self.pos = pos
        self.radius = radius
        self.startTheta = startTheta
        self.deltaTheta = deltaTheta
        self.width = width 

    def theta(self, xy):
        dx = xy[0]-self.pos[0]
        dy = xy[1]-self.pos[1]
        return math.atan(dy/dx)*180./math.pi

    def as_patch(self, **kwa):
        st = self.startTheta
        dt = self.deltaTheta
        w = self.width  

        if st is None and dt is None:
            return [mpatches.Circle(self.pos,self.radius, linewidth=w, **kwa)]
        elif st is None and dt is not None:
            log.info("wedge %s : %s " % (-dt, dt))
            return [mpatches.Wedge(self.pos,self.radius,-dt, dt, width=w, **kwa)] 
        else:
            log.info("wedges %s : %s " % (st, st+dt))
            log.info("wedges %s : %s " % (-st-dt, -st))
            return [mpatches.Wedge(self.pos,self.radius, st, st+dt, width=w, **kwa),
                    mpatches.Wedge(self.pos,self.radius, -st-dt, -st, width=w, **kwa)] 
 



class Chord(object):
    """
    Common Chord of two intersecting circles
    """
    def __init__(self, npos, ppos, a, b ):
        self.npos = npos
        self.ppos = ppos
        self.a = a 
        self.b = b  

    def as_lin(self): 
        return Lin(self.npos, self.ppos)

    def get_circ(self, c):
        nTheta = c.theta(self.npos) 
        pTheta = c.theta(self.ppos) 
        if nTheta > pTheta:
           sTheta, dTheta = pTheta, nTheta - pTheta
        else: 
           sTheta, dTheta = nTheta, pTheta - nTheta
        return Circ(c.pos, c.radius, sTheta, dTheta)
         
    def get_circ_a(self):
        return self.get_circ(self.a) 
    def get_circ_b(self):
        return self.get_circ(self.b) 


class Lin(object):
    def __init__(self, a, b ):
        self.a = a
        self.b = b 

    def as_line(self, lw=0.1, alpha=0.5):
        lx = [self.a[0], self.b[0]]
        ly = [self.a[1], self.b[1]]
        line = mlines.Line2D( lx, ly, lw=lw, alpha=alpha)
        return line


class Tub(object):
    def __init__(self, pos, radius, sizeZ):
        self.pos = pos
        self.radius = radius
        self.sizeZ = sizeZ

    def as_patch(self, **kwa):
        botleft = [ self.pos[0], self.pos[1] - self.radius ]
        width = self.sizeZ
        height = 2.*self.radius
        log.info("rect %s %s %s " % (botleft, width, height ))
        return [mpatches.Rectangle(botleft, width, height, **kwa)]


class RevPlot(object):
    def __init__(self, ax, revs):     

        edgecolor = ['r','g','b']

        zmin = 1e6
        zmax = -1e6
  
        ymin = 1e6
        ymax = -1e6

        circs = []
        rects = []
        lins = []
        patches = []
        kwa = {}

        for i,rev in enumerate(revs):
 
            kwa['edgecolor']=edgecolor[i%len(edgecolor)]
            kwa['facecolor']='w'
            kwa['alpha'] = 0.5 

            xyz,r,sz = rev.xyz, rev.radius, rev.sizeZ
            z = xyz[2]
            pos = (z,0.)

            if rev.typ == "Sphere":

                circ = Circ( pos,r,  rev.startTheta, rev.deltaTheta, rev.width) 
                circs.append(circ)
                patches.extend(circ.as_patch(**kwa))

                if z-r < zmin:
                    zmin = z-r
                if z+r > zmax:
                    zmax = z+r
                if -r < ymin:
                    ymin = -r
                if r > ymax:
                    ymax = r

            elif rev.typ == "Tubs":

                tub = Tub( pos, r, rev.sizeZ ) 
                patches.extend(tub.as_patch(**kwa))   
            pass
        pass

        for i in range(1,len(circs)):
            a = circs[i-1]
            b = circs[i]
            ch = Circ.intersect(a,b)
            if ch is not None:
                lins.append(ch.as_lin())
                wa = ch.get_circ_a() 
                patches.extend(wa.as_patch(**kwa))


        xlim = [zmin, zmax]
        ylim = [ymin, ymax]

        for p in patches:
            ax.add_artist(p)

        for i in range(len(lins)):
            a = 0.1 if i == 1 else 0.9
            ax.add_line(lins[i].as_line(alpha=a))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    
        self.xlim = xlim
        self.ylim = ylim 



def split_plot(g, lvns):
    lvns = lvns.split()
    xlim = None 
    ylim = None 
    nplt = len(lvns)
    if nplt == 1:
        nx, ny = 1, 1
    else:
        nx, ny = 2, 2

    fig = plt.figure()
    for i in range(nplt):
        lv = g.logvol_(lvns[i])
        revs = lv.allrev()
        print "\n".join(map(str,revs))

        ax = fig.add_subplot(nx,ny,i+1, aspect='equal')
        ax.set_title(lv.name)

        rp = RevPlot(ax,revs)

        if xlim is None:
            xlim = rp.xlim 
        else:
            ax.set_xlim(*xlim) 

        if ylim is None:
           ylim = rp.ylim 
        else:
            ax.set_ylim(*ylim) 

    pass 
    fig.show()
    fig.savefig("/tmp/pmt_split_plot.png")



def single_plot(g, lvns_):
    lvns = lvns_.split()
    revs = []
    for i in range(len(lvns)):
        lv = g.logvol_(lvns[i])
        revs += lv.allrev()
        print "\n".join(map(str,revs))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title(lvns_)
    rp = RevPlot(ax,revs)

    fig.show()
    fig.savefig("/tmp/pmt_single_plot.png")





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    g.dump_context('PmtHemi')

    #lvns = "lvPmtHemi lvPmtHemiVacuum lvPmtHemiCathode"
    #lvns = "lvPmtHemi"
    lvns = "lvPmtHemiVacuum"
    #lvns = "lvPmtHemiCathode"
    #lvns = "lvPmtHemiBottom"
    #lvns = "lvPmtHemiDynode"

    #split_plot(g, lvns)
    single_plot(g, lvns)


