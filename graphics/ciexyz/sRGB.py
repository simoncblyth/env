#!/usr/bin/env python
"""
sRGB.py
========

The sRGB class derives the 3x3 linear conversion matrices between sRGB and CIE XYZ
color spaces from first principals, staring solely from two 2D specs:

1. chromaticity coordinates of the sRGB primaries (Red, Green, Blue)
2. standard D65 white point

What is Chromaticity ?
-----------------------

Chromaticity is the pure quality of a color (hue and saturation) completely
separated from its brightness(luminance).

From 3D XYZ to 2D Chromaticity (xy)
-----------------------------------

In the 1931 CIE XYZ space:

* Y represents luminance (perceived lightness)
* X,Z carry color information

Mapping from X,Y,Z to normalized x,y,z with:


        X                  Y                    Z
x = ------------,  y = ------------,   z =  ----------- ,   x + y + z = 1
     X + Y + Z          X + Y + Z            X + Y + Z

Due to the normalization one of x,y,z is redundant. Conventionally z.

A plot of x vs y is the 2D CIE 1931 chromaticity diagram

* outer boundary (spectral locus) - pure monochromatic wavelengths
* inner area - all colors that human eyes can perceive





"""
import os, logging
log = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt


class sRGB(object):
    """
    Getting to grips with color spaces by obtaining the
    sRGB to XYZ matrices from the chromaticity of R,G,B primaries
    and white point

    The transformations between XYZ and linear RGB are
    below obtained following technique from

    http://www.ryanjuckett.com/programming/rgb-color-space-conversion/
    """

    # CIE 1931 (x,y) chromaticity targets for sRGB
    xy = np.array([
             [0.3127, 0.3290],     # White Point (D65)
             [0.64,0.33],          # Red Primary
             [0.30,0.60],          # Green Primary
             [0.15,0.06]           # Blue Primary
             ])

    names = ["W", "R","G","B"]

    def plot(self):
        xyz = self.xyz
        plt.scatter( xyz[:,0], xyz[:,1] )
        for i in range(len(self.names)):
            plt.annotate(self.names[i], xy = xyz[i][:2], xytext = (0.5, 0.5), textcoords = 'offset points')
        pass
        plt.show()


    def __init__(self):
        """

        ::

            In [4]: wpxyz = np.array([0.3127,0.3290,0.3583])   ## 1-0.3127-0.3290 =  0.3583

            In [5]: wpxyz/wpxyz[1]
            Out[5]: array([0.95045593, 1.        , 1.08905775])

        """
        # for the 4 targets WRGB : construct xyz from xy by adding the redundant third column using  z=1-x-y
        xy = self.xy

        xyz = np.empty((len(xy),3))
        xyz[:,:2] = xy
        xyz[:,2] = np.ones_like(xyz[:,2]) - np.sum(xyz[:,:2], axis=1)

        # white point xyz coordinate to an XYZ coordinate by using a Y luminance value of 1.
        wpY = xyz[0][1]    # 0.32900000
        wXYZ = xyz[0]/wpY    # scale to make white point Y luminance of 1: [ 0.95 ,  1.   ,  1.089]

        # White Point defined as (R,G,B) = (1,1,1), luminance Y = 1
        #
        #
        # Relationship between linear (R,G,B) and the absolute (X,Y,Z)
        # is expressed using the (x,y,z) of values of the primaries
        #
        #   |X|     | xr xg xb | | CR  0   0 | |R|
        #   |Y|  =  | yr yg yb | | 0  CG   0 | |G|
        #   |Z|     | zr zg zb | | 0   0  CB | |B|
        #
        #
        #    (xr,yr,zr) are the (x,y,z) of the Red   primary
        #    (xg,yg,zg) are the (x,y,z) of the Green primary
        #    (xb,yb,zb) are the (x,y,z) of the Blue  primary
        #         these represent the direction of the primaries in xyz space 
        #
        #  Plugging the white point, namely (R,G,B) = (1,1,1)
        #  into the above - allows to obtain (CR,CG,CB) which
        #  is XYZSum
        #
        #
        # Solve for the (X+Y+Z) scalar values that will convert each xyz primary to XYZ space.
        # XYZSum is (CR,CG,CB)
        XYZSum = np.dot( np.linalg.inv(xyz[1:4].T) , wXYZ )

        # transposiin 


        # Reconstruct the matrix M which transforms from linear sRGB space to XYZ space.
        linear_sRGB_to_XYZ = np.dot( xyz[1:4].T, np.diag(XYZSum) )
        XYZ_to_linear_sRGB = np.linalg.inv(linear_sRGB_to_XYZ)

        self.xyz = xyz
        self.wXYZ = wXYZ
        self.XYZSum = XYZSum
        self.XYZ_to_linear_sRGB = XYZ_to_linear_sRGB
        self.linear_sRGB_to_XYZ = linear_sRGB_to_XYZ
        self.x2r = XYZ_to_linear_sRGB
        self.r2x = linear_sRGB_to_XYZ
        self.wpY = wpY

    @classmethod
    def check(cls):
        """
        Sanity check the matrix by conversion of RGB into XYZ
        """
        srgb = cls()

        w = [1,1,1]
        r = [1,0,0]
        g = [0,1,0]
        b = [0,0,1]

        wXYZ = np.dot( srgb.r2x, w )
        rXYZ = np.dot( srgb.r2x, r )
        gXYZ = np.dot( srgb.r2x, g )
        bXYZ = np.dot( srgb.r2x, b )

        wxyz = wXYZ/wXYZ.sum()
        rxyz = rXYZ/rXYZ.sum()
        gxyz = gXYZ/gXYZ.sum()
        bxyz = bXYZ/bXYZ.sum()

        log.info("w %15s wXYZ %20s wxyz %20s (expect) %20s " % (w,wXYZ,wxyz,srgb.xyz[0]))
        log.info("r %15s rXYZ %20s rxyz %20s (expect) %20s " % (r,rXYZ,rxyz,srgb.xyz[1]))
        log.info("g %15s gXYZ %20s gxyz %20s (expect) %20s " % (g,gXYZ,gxyz,srgb.xyz[2]))
        log.info("b %15s bXYZ %20s bxyz %20s (expect) %20s " % (b,bXYZ,bxyz,srgb.xyz[3]))

        assert np.allclose(srgb.xyz[0], wxyz)
        assert np.allclose(srgb.xyz[1], rxyz)
        assert np.allclose(srgb.xyz[2], gxyz)
        assert np.allclose(srgb.xyz[3], bxyz)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)


    sRGB.check()
    srgb = sRGB()
    if "PLOT" in os.environ:srgb.plot()

    xyz = srgb.xyz
    wXYZ = srgb.wXYZ
    XYZSum = srgb.XYZSum
    XYZ_to_linear_sRGB = srgb.XYZ_to_linear_sRGB
    linear_sRGB_to_XYZ = srgb.linear_sRGB_to_XYZ
    wpY = srgb.wpY






    from RGB import RGB

    rgb = RGB()
    rgb.table()

    assert (srgb.x2r-rgb.x2r).max() < 0.001
    assert (srgb.x2r-rgb.x2r).min() > -0.001




