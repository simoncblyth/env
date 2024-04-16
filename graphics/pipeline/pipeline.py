#!/usr/bin/env python
"""
~/env/graphics/pipeline/pipeline.py
===================================

* http://www.songho.ca/opengl/gl_projectionmatrix.html


::

   glm- ; vim -R $(glm-prefix)/glm/ext/matrix_clip_space.inl


Need projection matrix to do two things

1. map truncated pyramid frustum to unit cube 
2. tee up the perspective divide by the w homogenous coordinate



* https://docs.sympy.org/latest/modules/matrices/matrices.html

::


    In [11]: from sympy import simplify 

    In [12]: simplify(C)
    Out[12]: 
    [  2                    l + r]
    [------    0       0    -----]
    [-l + r                 l - r]
    [                            ]
    [          2            b + t]
    [  0     ------    0    -----]
    [        -b + t         b - t]
    [                            ]
    [                 -2    f + n]
    [  0       0     -----  -----]
    [                f - n  f - n]
    [                            ]
    [  0       0       0      1  ]


    In [19]: simplify(B*A*P)
    Out[19]: 
    [ 2*n              l + r         ]
    [------    0       -----      0  ]
    [-l + r            l - r         ]
    [                                ]
    [         2*n      b + t         ]
    [  0     ------    -----      0  ]
    [        -b + t    b - t         ]
    [                                ]
    [                -(f + n)   2*f*n]
    [  0       0     ---------  -----]
    [                  f - n    f - n]
    [                                ]
    [  0       0         1        0  ]

"""

from sympy import Matrix, symbols, simplify

from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)

ex,ey,ez = symbols("ex ey ez")
lx,ly,lz = symbols("lx ly lz")
ux,uy,uz = symbols("ux uy uz")




n,f,t,b,l,r = symbols("n f t b l r")

# translate the center of the frustum to the origin 
A = Matrix([
        [1,0,0,-(l+r)/2],
        [0,1,0,-(b+t)/2],
        [0,0,1,-(n+f)/2],
        [0,0,0,1]
      ])


# scale to unit cube 
B = Matrix([
       [2/(r-l),0      ,0      ,0],
       [0,      2/(t-b),0      ,0],
       [0,      0      ,2/(n-f),0],
       [0,      0      ,0      ,1]
     ])

BA = B*A     # hmm opposite order to normal as are using transposed form compared to OpenGL 
print(BA)

P = Matrix([
      [n,   0,    0,    0],
      [0,   n,    0,    0],
      [0,   0,    n+f,  -f*n],
      [0,   0,    1,    0]
    ])


"""
P is verbatim from ~/opticks_refs/INFOGR_2012-2013_lecture-07_projection.pdf
but I think this gets a subtlety wrong 

n and f are defined to be positive but the near and far planes 
are along -ve Z axis in eye frame 

    z = -n 
    z = -f 


So to fix this make a switch n->(-n) and f->(-f)
and then multiply by -1 
(as projection matrix is degenerate to a constant factor)
"""

N = Matrix([
      [-n,  0,    0,     0],
      [0,  -n,    0,     0],
      [0,   0,  -(n+f), -f*n],
      [0,   0,    1,     0]
    ])




BAP = simplify(B*A*P)
BAN = simplify(B*A*N)

print(simplify(BAP))




