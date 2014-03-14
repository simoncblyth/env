#!/usr/bin/env python
"""

"""
import numpy as np

def canonical_to_screen( nx, ny, debug=True, flip=False ):
    """
    Canonical::

                1
                |   
                |   
         -1-----o-----1
                |     
                |
               -1


    Non flipped screen::

                   (nx,ny)
         |         *
         |
         |         
         |         
         o---------


    Flipped::

         o---------
         |
         |
         | 
         |        * (nx,ny)



    Map square [-1,1]x[-1,1] onto rectange [0,nx]x[0,ny]
    """
    # scale -1:1 onto 0:nx 0:ny 
    s = np.identity(4)
    s[0,0] = nx/2.
    s[1,1] = ny/2.
    s[2,2] = 1.
 
    # translate scaled center to put 0,0 at bottom left
    t = np.identity(4)
    t[:3,3] = np.array([nx,ny,0])/2.

    # scale before translate, as translation was in terms of the scaled
    m = np.dot(t,s)

    # flip to get screen pixel (0,0) at(left,top) rather than (left,bottom) 
    if flip:
        m[1,1] = -m[1,1]

    if debug: 
        print "t\n", t
        print "s\n", s
        print "m\n", m

    return m



def test_canonical_to_screen():
    width, height, flip, debug = 640, 480, True, True

    m = canonical_to_screen(width, height, debug=debug, flip=flip)

    # canonical XY coords
    lt = [-1, 1,0]
    lb = [-1,-1,0]
    rt = [1,1,0]
    rb = [1,-1,0]
    ce = [0,0,0]

    # expectations
    if flip:
        x_lt = [0,0,0]
        x_lb = [0,height,0]
        x_rt = [width,0,0]
        x_rb = [width,height,0]
        x_ce = [width/2., height/2.,0]
    else:
        x_lt = [0,height,0]
        x_lb = [0,0,0]
        x_rt = [width,height,0]
        x_rb = [width,0,0]
        x_ce = [width/2., height/2.,0]

    # screen coords
    s_lt = np.dot(m, np.append(lt,1))
    s_lb = np.dot(m, np.append(lb,1))
    s_rt = np.dot(m, np.append(rt,1))
    s_rb = np.dot(m, np.append(rb,1))
    s_ce = np.dot(m, np.append(ce,1))

    assert np.allclose(np.append(x_lt,1),s_lt),(x_lt,s_lt)
    assert np.allclose(np.append(x_lb,1),s_lb),(x_lb,s_lb)
    assert np.allclose(np.append(x_rt,1),s_rt),(x_rt,s_rt)
    assert np.allclose(np.append(x_rb,1),s_rb),(x_rb,s_rb)
    assert np.allclose(np.append(x_ce,1),s_ce),(x_ce,s_ce)

    if debug:
        print "lt %-15s s_lt %-15s " % ( lt, s_lt )
        print "lb %-15s s_lb %-15s " % ( lb, s_lb )
        print "rt %-15s s_rt %-15s " % ( rt, s_rt )
        print "rb %-15s s_rb %-15s " % ( rb, s_rb )
        print "ce %-15s s_ce %-15s " % ( ce, s_ce )



if __name__ == '__main__':
    test_canonical_to_screen()


