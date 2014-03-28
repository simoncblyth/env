#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import math
import numpy as np


class DAECamera(object):
    def __init__(self, size=(640,480), yfov=50., near=10., far=20000. , nearclip=(1e-6,1e6), farclip=(1e-6,1e6), yfovclip=(1.,179)): 

        self.size = size
        self._yfov = yfov
        self._near = near
        self._far = far

        self.yfovclip = yfovclip   # extreme angles are handy in parallel projection
        self.nearclip = nearclip
        self.farclip = farclip

    def resize(self, size):
        self.size = size
        log.info("%s resize %s " % (self.__class__.__name__, str(size) ))
    aspect = property(lambda self:float(self.size[0])/float(self.size[1]))

    def __repr__(self):
        return "C %3.1f/%10.5f/%4.1f " % ( self._yfov, self._near, self._far )
    __str__ = __repr__


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


if __name__ == '__main__':
    pass
