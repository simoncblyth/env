#!/usr/bin/env python
"""

View frustum, with camera at origin looking in -Z direction, 
(the + should line up and point at origin, top view 
with Y out of page)::

                                                  +
                                    _             |
                                    |             |
                           + - - -  x             | 
                           | xs     |             |
       Z ------0 . . . . . |._ . . ._ . . . . . . |
               |           |        |             |
               |           +        |             |
               |         near       |             | 
               X                                  | 
                                                  +
                                                 far 
               -----d----->
               -------------z------->


      xs        x         (similar triangles)
     ----  =  ----
      d        -z

        
      xs   =   x d         (analogous for y, ys)
              --- 
               -z

           =   x n         (at the near plane, d = -n )
               ----
                z


Orthographic view volume::

                                                    
                           +----------------------+                           
                           |                      |
       Z ------0 . . . . . |. . . . . . . . . . . |
               |           |                      |
               |           +----------------------+                    
               |          near                    far       
               X                                     
  


Transformation needs to:

#. map lines through origin into lines 



"""

import numpy as np

def camera_to_orthographic(n, f):
    """


          xh' = x n

          yh' = y n 

          zh' = (n+f)z  - 2 fn 
               -----    -------
               (n-f)     (n-f) 

          wh' = z 


    Perspective divide by z to de-homogenize:


           x' =  x n 
                -----
                  z

           y' =  y n
                -----
                  z

           z' =   (n+f)       2nf         (at z=n,  z' = z) 
                  -----  -   ------       (at z=f,  z' = 
                  (n-f)      (n-f)z

           w' =  1 



    At z = n 


          x' =   x

          y' =   y  

          z' =    n+f-2f    = 1     
                 --------
                    n-f

    at z = f


          x' = x n
              -----
                f

          y' = y n
              -----
                f

          z' =   n+f  -  2n    = -1
                -------------
                    n-f




    """
    p0 = np.array((
          (  n,  0,   0,    0),
          (  0,  n,   0,    0),
          (  0,  0, n+f, -f*n),
          (  0,  0,   1,    0)
                              ),dtype='float')


    p1 = np.array((
          (  n,  0,           0,           0),
          (  0,  n,           0,           0),
          (  0,  0, (n+f)/(n-f), -2*f*n/(n-f)),
          (  0,  0,           1,            0)
                                        ),dtype='float')


    #print p
    return p0




def orthographic_projection( left, right, top, bottom, near, far ):
    """
    http://en.wikipedia.org/wiki/Orthographic_projection_(geometry)
    """
    s = np.identity(4)
    s[0,0] = 2./(right-left)
    s[1,1] = 2./(top-bottom)
    s[2,2] = 2./(far-near)

    center = np.array([left+right,top+bottom,far+near])/2. 
    t = np.identity(4)
    t[:3,3] = -center
    t[2,2] = -1        # Z flip 

    m = np.dot(s, t)
    return m 



if __name__ == '__main__':
    pass
    near, far = 1., 100.
    pers = camera_to_orthographic( near, far )

    L = 1
    ez_ = lambda z:near + far - (far*near/z) 

    nl = [-L, 0, near]
    nc = [0, 0, near]
    nr = [L, 0, near]
    ne = ez_(near)

    fl = [-L*far/near, 0, far]
    fc = [          0, 0, far]
    fr = [L*far/near, 0, far]
    fe = ez_(far) 

    pos = (far+near)/2. 
    pl = [-L*pos/near, 0, pos]
    pc = [          0, 0, pos]
    pr = [L*pos/near, 0, pos]
    pe = ez_(pos) 


    def t_( m , v ):
        p = np.dot( m, np.append(v,1))
        p /= p[3]
        return p 


    p_nl = t_( pers , nl )
    p_nc = t_( pers , nc )
    p_nr = t_( pers , nr )

    print "nl %-15s  p_nl %-15s " % ( nl, p_nl ) 
    print "nc %-15s  p_nc %-15s " % ( nc, p_nc ) 
    print "nr %-15s  p_nr %-15s " % ( nr, p_nr ) 
    print "ne %-15s " % ne


    p_fl = t_( pers , fl )
    p_fc = t_( pers , fc )
    p_fr = t_( pers , fr )

    print "fl %-15s  p_fl %-15s " % ( fl, p_fl ) 
    print "fc %-15s  p_fc %-15s " % ( fc, p_fc ) 
    print "fr %-15s  p_fr %-15s " % ( fr, p_fr ) 
    print "fe %-15s " % fe


    p_pl = t_( pers , pl )
    p_pc = t_( pers , pc )
    p_pr = t_( pers , pr )

    print "pl %-15s  p_pl %-15s " % ( pl, p_pl ) 
    print "pc %-15s  p_pc %-15s " % ( pc, p_pc ) 
    print "pr %-15s  p_pr %-15s " % ( pr, p_pr ) 
    print "pe %-15s " % pe






