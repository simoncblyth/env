#!/usr/bin/env python
"""
~/env/graphics/pipeline/pipeline.py
===================================

* http://www.songho.ca/opengl/gl_projectionmatrix.html
* https://docs.sympy.org/latest/modules/matrices/matrices.html

::

   glm- ; vim -R $(glm-prefix)/glm/ext/matrix_clip_space.inl



glm/glm/detail/setup.hpp::

     477 ///////////////////////////////////////////////////////////////////////////////////
     478 // Clip control, define GLM_FORCE_DEPTH_ZERO_TO_ONE before including GLM
     479 // to use a clip space between 0 to 1.
     480 // Coordinate system, define GLM_FORCE_LEFT_HANDED before including GLM
     481 // to use left handed coordinate system by default.
     482 
     483 #define GLM_CLIP_CONTROL_ZO_BIT     (1 << 0) // ZERO_TO_ONE
     484 #define GLM_CLIP_CONTROL_NO_BIT     (1 << 1) // NEGATIVE_ONE_TO_ONE
     485 #define GLM_CLIP_CONTROL_LH_BIT     (1 << 2) // LEFT_HANDED, For DirectX, Metal, Vulkan
     486 #define GLM_CLIP_CONTROL_RH_BIT     (1 << 3) // RIGHT_HANDED, For OpenGL, default in GLM
     487 
     488 #define GLM_CLIP_CONTROL_LH_ZO (GLM_CLIP_CONTROL_LH_BIT | GLM_CLIP_CONTROL_ZO_BIT)
     489 #define GLM_CLIP_CONTROL_LH_NO (GLM_CLIP_CONTROL_LH_BIT | GLM_CLIP_CONTROL_NO_BIT)
     490 #define GLM_CLIP_CONTROL_RH_ZO (GLM_CLIP_CONTROL_RH_BIT | GLM_CLIP_CONTROL_ZO_BIT)
     491 #define GLM_CLIP_CONTROL_RH_NO (GLM_CLIP_CONTROL_RH_BIT | GLM_CLIP_CONTROL_NO_BIT)
     492 
     493 #ifdef GLM_FORCE_DEPTH_ZERO_TO_ONE
     494 #   ifdef GLM_FORCE_LEFT_HANDED
     495 #       define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_LH_ZO
     496 #   else
     497 #       define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_RH_ZO
     498 #   endif
     499 #else
     500 #   ifdef GLM_FORCE_LEFT_HANDED
     501 #       define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_LH_NO
     502 #   else
     503 #       define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_RH_NO
     504 #   endif
     505 #endif

     Implies default GLM_CONFIG_CLIP_CONTROL is GLM_CLIP_CONTROL_RH_NO
     meaning RH and -1 to 1 clip space. 

glm/ext/matrix_clip_space.inl::

    209     template<typename T>
    210     GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustum(T left, T right, T bottom, T top, T nearVal, T farVal)
    211     {
    212         if(GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_ZO)
    213             return frustumLH_ZO(left, right, bottom, top, nearVal, farVal);
    214         else if(GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_LH_NO)
    215             return frustumLH_NO(left, right, bottom, top, nearVal, farVal);
    216         else if(GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_ZO)
    217             return frustumRH_ZO(left, right, bottom, top, nearVal, farVal);
    218         else if(GLM_CONFIG_CLIP_CONTROL == GLM_CLIP_CONTROL_RH_NO)
    219             return frustumRH_NO(left, right, bottom, top, nearVal, farVal);
    220     }


    159     template<typename T>
    160     GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumRH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
    161     {
    162         mat<4, 4, T, defaultp> Result(0);
    163         Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
    164         Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
    165         Result[2][0] = (right + left) / (right - left);
    166         Result[2][1] = (top + bottom) / (top - bottom);
    167         Result[2][2] = - (farVal + nearVal) / (farVal - nearVal);
    168         Result[2][3] = static_cast<T>(-1);
    169         Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
    170         return Result;
    171     }
 
    Copying from above frustumRH_NO

    |    2n/(r-l)    0           (r+1)/(r-1)     0           |
    |                                                        |
    |     0        2n/(t - b)    (t+b)/(t-b)     0           | 
    |                                                        |  
    |     0          0          -(f+n)/(f-n)  -2.*f*n/(f-n)  |  
    |                           ^                            | 
    |     0          0               -1          0           |
                                     ^ 

    This matches the documented matrix from the below

    * https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml

    (left, bottom, -near)   ==>   lower left of window  (-1,-1)
    (right,  top,  -far)    ==>   upper right of window (+1,+1)




    131     template<typename T>
    132     GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> frustumLH_NO(T left, T right, T bottom, T top, T nearVal, T farVal)
    133     {
    134         mat<4, 4, T, defaultp> Result(0);
    135         Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
    136         Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
    137         Result[2][0] = (right + left) / (right - left);
    138         Result[2][1] = (top + bottom) / (top - bottom);
    139         Result[2][2] = (farVal + nearVal) / (farVal - nearVal);
    140         Result[2][3] = static_cast<T>(1);
    141         Result[3][2] = - (static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
    142         return Result;
    143     }

   Copying from above frustumLH_NO

    |    2n/(r-l)    0           (r+1)/(r-1)     0           |
    |                                                        |
    |     0        2n/(t - b)    (t+b)/(t-b)     0           | 
    |                                                        |  
    |     0          0           (f+n)/(f-n)  -2.*f*n/(f-n)  |  
    |                           ^                             | 
    |     0          0               1           0           |
                                    ^  







Need projection matrix to do two things

1. map truncated pyramid frustum to unit cube 
2. tee up the perspective divide by the w homogenous coordinate


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


"""
Q: Are n and f positive such that near/far planes are at z = -n and z = -f ?

"""

n,f,t,b,l,r = symbols("n f t b l r")

A_label = "translate the center of the frustum to the origin" 
A = Matrix([
        [1,0,0,-(l+r)/2],
        [0,1,0,-(b+t)/2],
        [0,0,1,-(n+f)/2],
        [0,0,0,1]
      ])


B_label = "scale to unit cube" 
B = Matrix([
       [2/(r-l),0      ,0      ,0],
       [0,      2/(t-b),0      ,0],
       [0,      0      ,2/(f-n),0],     
       [0,      0      ,0      ,1]
     ])

## that was n-f

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


Z = Matrix([
      [1,   0,    0,     0],
      [0,   1,    0,     0],
      [0,   0,   -1,     0],
      [0,   0,    0,     1]
    ])




print("simplify(B*A*P)")
print(simplify(B*A*P))

print("simplify(B*A*Z*P)")
print(simplify(B*A*Z*P))




#BAN = simplify(B*A*N)
#print("simplify(BAN)")
#print(simplify(BAN))




