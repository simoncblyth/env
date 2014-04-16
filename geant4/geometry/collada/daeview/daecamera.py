#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import math
import numpy as np


class DAECamera(object):
    def __init__(self, size=(640,480), kscale=1., yfov=50., near=10., far=20000. , nearclip=(1e-6,1e6), farclip=(1e-6,1e6), yfovclip=(1.,179)): 

        self.size = np.array(size, dtype=int )
        self.kscale = kscale   # i dont like it, but better that putting in the view or scene

        self._yfov = yfov
        self._near = near
        self._far = far

        self.yfovclip = yfovclip   # extreme angles are handy in parallel projection
        self.nearclip = nearclip
        self.farclip = farclip

    def resize(self, size):
        self.size = size
        log.debug("%s resize %s " % (self.__class__.__name__, str(size) ))

    aspect = property(lambda self:float(self.size[0])/float(self.size[1]))

    def smry(self):
        return "\n".join([
                      "size        %s " % self.size, 
                      "kscale      %s " % self.kscale, 
                      "aspect      %s " % self.aspect, 
                      "yfov        %s " % self.yfov, 
                      "near        %s " % self.near, 
                      "far         %s " % self.far,
                      "lrbtnf      %s " % self.lrbtnf,
                      "nearsize    %s " % self.nearsize,
                      "farsize     %s " % self.farsize,
                      "pixelsize   %s " % self.pixelsize,
                      "pixelhalf   %s " % self.pixelhalf,
                        ]) 

    def __repr__(self):
        return "C %3.1f/%10.5f/%4.1f " % ( self._yfov, self._near, self._far )

    def __str__(self):
        ii_ = lambda name:"--%(name)s=%(fmt)s,%(fmt)s" % dict(fmt="%d",name=name) 
        f_ = lambda name,fmt:"--%(name)s=%(fmt)s" % dict(fmt=fmt,name=name) 
        return   " ".join(map(lambda _:_.replace(" ",""),[
                         ii_("size")  % self.size,
                         f_("near","%10.5f")  % self.near,
                         f_("far","%4.1f")  % self.far,
                         f_("yfov","%3.1f")  % self.yfov,
                          ])) 

    def near_to (self, x, y, dx, dy):
        ''' Change near clipping '''
        self.near += self.near*dy

    def far_to (self, x, y, dx, dy):
        ''' Change far clipping '''
        self.far += self.far*dy

    def yfov_to (self, x, y, dx, dy):
        ''' Change yfov '''
        self.yfov += 50*dy

    def _get_near(self):
        return self._near
    def _set_near(self, near):
        self._near = np.clip(near, self.nearclip[0], self.nearclip[1])
    near = property(_get_near, _set_near)

    def _get_far(self):
        return self._far
    def _set_far(self, far):
        self._far = np.clip(far, self.farclip[0],self.farclip[1])
    far = property(_get_far, _set_far)

    def _get_yfov(self):
        return self._yfov
    def _set_yfov(self, yfov):
        self._yfov = np.clip(yfov,self.yfovclip[0],self.yfovclip[1])
    yfov = property(_get_yfov, _set_yfov)


    def _get_nearsize(self):
        height = 2. * self._near * math.tan(self._yfov*0.5*math.pi/180.0)  # symmetrical 
        return np.array([self.aspect*height, height], dtype=float )
    nearsize = property(_get_nearsize, doc="width, height at near plane")     

    def _get_farsize(self):
        nearsize = self.nearsize
        farnear = self._far/self._near 
        return np.array([farnear*nearsize[0], farnear*nearsize[1]], dtype=float)
    farsize = property(_get_farsize , doc="width, height at far plane ")     

    def _get_pixelsize(self):
        return self.nearsize/self.size   # elementwise numpy division
    pixelsize = property(_get_pixelsize , doc="width, height of single pixel at near plane ")     
    pixelhalf = property(lambda self:self.pixelsize/2.)     

    npixel = property(lambda self:self.size[0]*self.size[1])

    def pixel_index(self, xyzw ):
        """
        :param px:  pixel coordinates (0,0) at top left  (nx-1,ny-1) at bottom right
        :param py:
        """
        nx, ny = self.size            # size in pixels
        return xyzw[1] * nx + xyzw[0]

    def pixel_xyzw(self, index ):
        """
        :param index: pixel index
        :return: x,y integer pixel coordinates,  raster style (0,0) at top left
        """
        nx, ny = self.size 
        assert 0 <= index < nx*ny        
        return np.array([ index % nx, index / nx , 0, 1], dtype=int)

    def _get_pixel2camera(self):
        """
        Consider a 4x4 grid::

             |0|1|2|3|
             |4|5|6|7|
             |8|9|A|B|
             |C|D|E|F|
             
        #. pixel 0 at (0,0) 
        #. pixel F at (3,3) (nx-1, ny-1) 
        #. regarding grid size as  (nx-1-0,ny-1-0)  misses half pixels in each direction

        CAUTION: y flip in pixel numbering compared to glReadPixels

        * https://www.opengl.org/sdk/docs/man2/xhtml/glReadPixels.xml

        glReadPixels numbers pixels with lower left corner at (x+i,y+j )
        for 0 <= i < width and 0 <= j < height 


        Image square::

            l,t ________ r,t
               |        |
               |        |
               |________| 
            l,b          r,b
                 
        Mapping to the below looses a pixel, half pixel in each direction::

              [  0 , nx - 1 ] x [ 0 ,  ny - 1  ]

        Treating pixels to represent unit squares centered at integer coordinates,
        need to grow by 1/2 pixel in each direction, thus mapping to::

              [ -1/2 , nx - 1  + 1/2 ] x [ -1/2,  ny - 1 + 1/2 ]
              [ -1/2 , nx - 1/2 ] x [ -1/2,  ny - 1/2 ]

        """ 
        nx, ny = self.size            # size in pixels
        l,r,b,t,n,f = self.lrbtnf

        pixel_offset=(-0.5,-0.5,0)
        pixel_translate = np.array([nx/2., ny/2.,0]) + np.array(pixel_offset)

        mt = np.identity(4)
        mt[:3,3] = -pixel_translate   # centers pixels

        ms = np.identity(4)
        ms[0,0] = 2./nx
        ms[1,1] = 2./ny               # scale pixels down to -1:1

        mu = np.identity(4)
        mu[0,0] = (r - l)/2.
        mu[1,1] = (t - b)/2.          # scale to eye coordinate dimensions

        mz = np.identity(4)
        mz[2,3] = -self.near          # z translate 

        zust = np.dot( mz, np.dot( mu, np.dot( ms, mt ))) 
        return zust
 
    pixel2camera = property(_get_pixel2camera, doc="4x4 matrix transforming raster style pixel coordinates into eye positions placed at z=-near" )


    def pixel_pos(self, index): 
        """
        :param index: pixel index, where index 0 corresponds to raster pixel (0,0) at top left
        :return xyz:  eye frame coordinate of pixel

        ::

            In [46]: camera.pixel_pos(camera.npixel-1)
            Out[46]: array([  6.208,   4.653, -10.   ,   1.   ])

            In [48]: camera.pixel_pos(0)                         # huh, where did the yflip get done ?
            Out[47]: array([ -6.208,  -4.653, -10.   ,   1.   ])


        """
        pxyzw = self.pixel_xyzw( index )
        pixel2camera = self.pixel2camera

        return pixel2camera.dot( pxyzw )

 
    def _get_lrbtnf(self):
        """
        ::

                   . | 
                .    | top 
              +------- 
                near |
                     |
                   
        """
        aspect = self.aspect
        near = self._near  
        far = self._far    
        top = near * math.tan(self._yfov*0.5*math.pi/180.0)  
        bottom = -top
        left = aspect * bottom
        right = aspect * top 

        return np.array([left,right,bottom,top,near,far]) 

    lrbtnf = property(_get_lrbtnf)

    def _get_pixel_corners(self):
        """

                   "xlt":(-0.5,-0.5),    
                   "xrb":(nx-1+.5,ny-1+.5), 
        """
        nx, ny = self.size
        corners = { 
                    "lt":(0,0,0,1)      ,      "mt":(nx/2-1,0,0,1)      , "rt":(nx-1,0,0,1)      ,
                    "lm":(0,ny/2-1,0,1) ,      "mm":(nx/2-1,ny/2-1,0,1) , "rm":(nx-1,ny/2-1,0,1) ,
                    "lb":(0,ny-1,0,1)   ,      "mb":(nx/2-1,ny-1,0,1)   , "rb":(nx-1,ny-1,0,1)   , 
                 } 
        return corners
    pixel_corners = property(_get_pixel_corners)



def check_pixel_index(camera):

    nx, ny = camera.size

    lt = (0,0)
    rt = (nx-1,0)

    lb = (0, ny-1)
    rb = (nx-1,ny-1)

    ilt = camera.pixel_index(lt)
    elt = 0
    assert ilt == elt  

    irt = camera.pixel_index(rt)
    ert = nx - 1
    assert irt == ert  

    ilb = camera.pixel_index(lb)
    elb = nx*ny - 1 - (nx - 1) 
    assert ilb == elb 

    irb = camera.pixel_index(rb)
    erb =  nx*ny - 1
    assert irb == erb, ("irb", irb, "erb", erb )

def check_pixel_xyzw(camera):
    nx, ny = camera.size
    index = 0 
    for iy in range(ny):
        for ix in range(nx): 
            px, py, pz, pw = camera.pixel_xyzw(index)
            index += 1
            assert (px,py,pz,pw) == (ix,iy,0,1) 

def check_pixel_pos(camera):
    nx, ny = camera.size
    xindex = 0
    for iy in range(ny):
        for ix in range(nx):
            pindex = camera.pixel_index((ix,iy))
            assert xindex == pindex , (xindex, pindex, ix, iy, nx, ny  )

            pxyzw = camera.pixel_xyzw(pindex)
            assert np.all(pxyzw == (ix, iy, 0, 1)) 
            xindex += 1 

def check_expectations( label, v , e ):
    assert np.allclose(v,e), "%s = %s v %s e %s v-e %s " % ( label, np.allclose(v,e), str(v), str(e), str(v-e) ) 


def check_pixel_pos_corners(camera):

    nx,ny  = camera.size
    ph = np.append(camera.pixelhalf,[0,0])
    hx = np.array([ph[0],   0,0,0])
    hy = np.array([   0,ph[1],0,0])

    lt = (0,0)
    rt = (nx-1,0)

    lb = (0, ny-1)
    rb = (nx-1,ny-1)

    ilt = camera.pixel_index(lt)
    irt = camera.pixel_index(rt)
    ilb = camera.pixel_index(lb)
    irb = camera.pixel_index(rb)

    plt = camera.pixel_pos(ilt)
    prt = camera.pixel_pos(irt)
    plb = camera.pixel_pos(ilb)
    prb = camera.pixel_pos(irb)

    l,r,b,t,n,f = camera.lrbtnf

    # sign flip y, and half pixel offsets

    xlt = np.array([l,-t,-n,1]) + hx + hy 
    xrt = np.array([r,-t,-n,1]) - hx + hy 
    xlb = np.array([l,-b,-n,1]) + hx - hy 
    xrb = np.array([r,-b,-n,1]) - hx - hy 

    check_expectations( "lt", plt , xlt )     
    check_expectations( "rt", prt , xrt )     
    check_expectations( "lb", plb , xlb )     
    check_expectations( "rb", prb , xrb )     



if __name__ == '__main__':
    pass
    np.set_printoptions(precision=4, suppress=True)
    camera = DAECamera()
    print camera.smry()


    check_pixel_index(camera)
    check_pixel_xyzw(camera)
    check_pixel_pos(camera)
    check_pixel_pos_corners(camera)

    for label, pxyzw in sorted(camera.pixel_corners.items(), key=lambda kv:sum(kv[1])):
        index = camera.pixel_index(pxyzw)
        pos = camera.pixel_pos(index)
        print "%-5s %10s %10s %s " % ( label, pxyzw, index, pos )










