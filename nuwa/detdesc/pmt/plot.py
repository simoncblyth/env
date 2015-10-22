#!/usr/bin/env python
"""
Plotting from the serialized PMT analytic geometry data
"""
import numpy as np
import logging, os
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

path_ = lambda _:os.path.expandvars("$IDPATH/GMergedMesh/1/%s.npy" % _)

X = 0
Y = 1
Z = 2

ZX = [Z,X]
ZY = [Z,Y]
XY = [X,Y]


class Mesh(object):
    def __init__(self):
        self.v = np.load(path_("vertices"))
        self.f = np.load(path_("indices"))
        self.i = np.load(path_("nodeinfo"))
        self.vc = np.zeros( self.i.shape[0]+1 )
        np.cumsum(self.i[:,1], out=self.vc[1:])
    def verts(self, solid):
        return self.v[self.vc[solid]:self.vc[solid+1]]



class Sphere(object):
    def __init__(self, center, radius):
        self.center = center 
        self.radius = radius 
    def __repr__(self):
        return "Sphere %s %s  " % (repr(self.center), self.radius)

    def as_patch(self, axes):
        circle = mpatches.Circle(self.center[axes],self.radius)
        return circle 

class ZTubs(object):
    def __init__(self, position, radius, sizeZ):
        self.position = position
        self.radius = radius 
        self.sizeZ = sizeZ 
    def __repr__(self):
        return "ZTubs %s %s %s " % (repr(self.position), self.radius, self.sizeZ)

    def as_patch(self, axes):
        if Z == axes[0]:
            width = self.sizeZ
            height = 2*self.radius
            botleft = self.position[axes] - np.array([0, self.radius])
            patch = mpatches.Rectangle(botleft, width, height)
        elif Z == axes[1]:
            assert 0
        else:
            patch = mpatches.Circle(self.position[axes],self.radius)
        return patch


class Bbox(object):
    def __init__(self, min_, max_ ):
        self.min_ = np.array(min_)
        self.max_ = np.array(max_)
        self.dim  = max_ - min_

    def as_patch(self, axes):
         width = self.dim[axes[0]]
         height = self.dim[axes[1]]
         botleft = self.min_[axes]
         rect = mpatches.Rectangle( botleft, width, height)
         return rect

    def __repr__(self):
        return "Bbox %s %s %s " % (self.min_, self.max_, self.dim )


class Pmt(object):
    def __init__(self, path):
        self.data = np.load(path).reshape(-1,4,4)
        self.num_parts = len(self.data)
        self.all_parts = range(self.num_parts)
        self.partcode = self.data[:,2,3].view(np.int32)
        self.partnode = self.data[:,3,3].view(np.int32)

    def parts(self, solid):
        """
        :param solid: index of node/solid 
        :return parts array:
        """
        return np.arange(len(self.partnode))[self.partnode == solid]

    def bbox(self, p):
        part = self.data[p]
        return Bbox(part[2,:3], part[3,:3])

    def shape(self, p):
        """
        :param p: part index
        :return shape instance: Sphere or ZTubs 
        """
        code = self.partcode[p]
        if code == 1:
            return self.sphere(p)
        elif code == 2:
            return self.ztubs(p)
        else:
            return None 

    def sphere(self, p):
        """
        Creates *Shape* instance from Part data identified by index

        :param p: part index
        :return Sphere:
        """
        part = self.data[p]
        return Sphere( part[0][:3], part[0][3])

    def ztubs(self, p):
        part = self.data[p]
        return ZTubs( part[0][:3], part[0][3], part[1][0])


class PmtPlot(object):
    def __init__(self, ax, pmt, axes):
        self.ax = ax
        self.axes = axes
        self.pmt = pmt
        self.patches = []
        self.ec = 'none'
        self.edgecolor = ['r','g','b','c','m','y','k']
        
    def plot_bbox(self, parts=[]):
        for p in parts:
            bb = self.pmt.bbox(p)
            _bb = bb.as_patch(self.axes)
            self.add_patch(_bb)

    def plot_shape(self, parts=[], clip=True):
        for i,p in enumerate(parts):
            self.ec = self.edgecolor[i%len(self.edgecolor)]
            bb = self.pmt.bbox(p)
            _bb = bb.as_patch(self.axes)
            self.add_patch(_bb)

            sh = self.pmt.shape(p)
            _sh = sh.as_patch(self.axes)
            self.add_patch(_sh)
            if clip:
                _sh.set_clip_path(_bb)

    def add_patch(self, patch):
        patch.set_fc('none')
        patch.set_ec(self.ec)
        self.patches.append(patch)
        self.ax.add_artist(patch)

    def limits(self, s):
        self.ax.set_xlim(-s,s)
        self.ax.set_ylim(-s,s)



def mug_plot(fig, pmt, solid, size):

    ax = fig.add_subplot(1,2,1, aspect='equal')
    pp = PmtPlot(ax, pmt, axes=ZX) 
    pp.plot_shape(pmt.parts(solid), clip=True)
    pp.limits(size)

    ax = fig.add_subplot(1,2,2, aspect='equal')
    pp = PmtPlot(ax, pmt, axes=XY) 
    pp.plot_shape(pmt.parts(solid), clip=True)
    pp.limits(size)



def one_plot(fig, pmt, solid, size, clip):

    ax = fig.add_subplot(1,1,1, aspect='equal')
    pp = PmtPlot(ax, pmt, axes=ZX) 
    pp.plot_shape(pmt.parts(solid), clip)
    pp.limits(size)


def clipped_unclipped_plot(fig, pmt, solid, size):

    ax = fig.add_subplot(1,2,1, aspect='equal')
    pp = PmtPlot(ax, pmt, axes=ZX) 
    pp.plot_shape(pmt.parts(solid), clip=False)
    pp.limits(size)

    ax = fig.add_subplot(1,2,2, aspect='equal')
    pp = PmtPlot(ax, pmt, axes=ZX) 
    pp.plot_shape(pmt.parts(solid), clip=True)
    pp.limits(size)


def one_plot_scatter(fig, pmt, solid, size, clip, axes, mesh):

    ax = fig.add_subplot(1,1,1, aspect='equal')
    pp = PmtPlot(ax, pmt, axes) 
    pp.plot_shape(pmt.parts(solid), clip)
    pp.limits(size)

    if mesh:
        vv = mesh.verts(solid)
        plt.scatter(vv[:,axes[0]],vv[:,axes[1]],c=vv[:,Y])



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    
    mesh = Mesh()

    pmt = Pmt("/tmp/hemi-pmt-parts.npy")
    fig = plt.figure()

    #mug_plot(fig, pmt, solid=0, size=150)
    #clipped_unclipped_plot(fig, pmt, solid=0, size=150)
    #one_plot(fig, pmt, solid=0, size=150, clip=True)
    one_plot_scatter(fig, pmt, solid=0, size=200, clip=False, axes=ZX, mesh=mesh)


    fig.show()
    fig.savefig("/tmp/plot.png")

