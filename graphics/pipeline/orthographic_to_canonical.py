#!/usr/bin/env python
"""
Orthographic to canonical
===========================

* Orthographic, axis aligned (l,b,n) (r,t,f)
* Canonical 2x2 box with extremities (-1,-1,1) (1,1,-1)   

* translation to get centered on (l+r,b+t,n+f)/2
* scaling 



::

               (1,1,-1)
               (r,t,f)      Y  -Z 
         +-----+            |  . 
        /|    /|            | .
       +-----+ |            |.
       | +   | +            +---- X
       |/    |/            /       
       +-----+            /
   (l,b,n)              +Z
   (-1,-1,1)

"""

import numpy as np

def orthographic_to_canonical( left, bottom, near, right, top, far, debug=False):

    center = np.array([left+right,bottom+top,near+far])/2.   

    if debug:
        print "left,right : ", left, right 
        print "top,bottom : ", top, bottom 
        print "near,far   : ", near, far
        print "center   : ", center

    # homogenous 4x4 matrix to translate center to origin
    t = np.identity(4)
    t[:3,3] = -center

    if debug: 
        print "t\n", t

    # scale onto -1 to 1 
    s = np.identity(4)
    s[0,0] = 2./(right-left)
    s[1,1] = 2./(top-bottom)
    s[2,2] = 2./(near-far)      # oops, this was sign flipped, but no impact as Z is about to be canned anyhow ?

    if debug:
        print "s\n", s

    # must translate before scaling 
    m = np.dot(s,t)

    if debug:
        print "m\n", m 
    
    return m 


def check_orthographic_to_canonical( lbn, rtf ):

    left, bottom, near = lbn
    right, top, far = rtf

    m = orthographic_to_canonical( left,bottom,near,right,top,far )

    cen = np.array([left+right,bottom+top,near+far])/2.

    lbn_o = np.dot(m,np.append(lbn,1))
    rtf_o = np.dot(m,np.append(rtf,1))
    cen_o = np.dot(m,np.append(cen,1))

    print "lbn_o ", lbn_o
    print "rtf_o ", rtf_o
    print "cen_o ", cen_o
 
    xlbn=(-1,-1,1)
    xrtf=(1,1,-1)
    xcen=(0,0,0)

    assert np.allclose( np.append(xlbn,1),lbn_o ),(xlbn, lbn_o)
    assert np.allclose( np.append(xrtf,1),rtf_o ),(xrtf, rtf_o)
    assert np.allclose( np.append(xcen,1),cen_o ),(xcen, cen_o)


def test_orthographic_to_canonical():
    aspect = 4./3.
    left,right = -10, 10
    bottom, top = left/aspect, right/aspect  
    near,far   = 1., 100.
    
    check_orthographic_to_canonical( (left,bottom,near),(right,top,far) )


def test_orthographic_to_canonical_diagonal():
    aspect = 4./3.
    left,right = -10, 10
    bottom, top = left/aspect, right/aspect  
    near,far   = 1., 100.
 
    m = orthographic_to_canonical( left,bottom,near,right,top,far )

    lbn = np.array([left,bottom,near])
    rtf = np.array([right,top,far])

    print "lbn", lbn
    print "rtf", rtf

    diag_ = lambda f:lbn*(1.-f) + rtf*f

    for f in (0,0.25,0.5,0.75,1.):
        dia = diag_(f)
        dia_o = np.dot(m,np.append(dia,1))
        xf = 2*f-1
        #print "diag      : %s    f %s " % (dia, f)
        print "diag_o    : %s    xf %s " % (dia_o, xf )
        xdia = np.array([xf,xf,-xf]) 
        assert np.allclose( np.append(xdia,1), dia_o ), (xdia, dia_o ) 




if __name__ == '__main__':
    pass
    test_orthographic_to_canonical()
    test_orthographic_to_canonical_diagonal()



 
