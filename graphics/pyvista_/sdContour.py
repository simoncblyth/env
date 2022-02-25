"""



"""

import numpy as np
import pyvista as pv

def sdf_sphere(x, y, z, radius=5):
    xx = x*x
    yy = y*y
    zz = z*z
    xx_yy_zz = xx + yy + zz
    return np.sqrt(xx_yy_zz) - radius 

def sdf_box(x,y,z, bx=1, by=2, bz=3):
    """

    float3 q = make_float3( fabs(pos.x) - q0.f.x/2.f, fabs(pos.y) - q0.f.y/2.f , fabs(pos.z) - q0.f.z/2.f ) ;
    float3 z = make_float3( 0.f );
    float sd = length(fmaxf(q, z)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f ) ;

    """
    qx = np.abs(x) - bx
    qy = np.abs(y) - by
    qz = np.abs(z) - bz

    px = np.maximum( qx, 0. )    
    py = np.maximum( qy, 0. )    
    pz = np.maximum( qz, 0. )    

    sd = np.sqrt( px*px + py*py + pz*pz )  + np.minimum( np.maximum.reduce([qx,qy,qz]), 0. )
    # 2nd term allows the sd to go negative inside the box
    return sd


def sdf_cylinder(x,y,z, z1=-5, z2=7, radius=5 ):
    """
    """
    sd_capslab = np.maximum( z - z2 , z1 - z ) 
    sd_infcyl = np.sqrt( x*x + y*y ) - radius 
    sd = np.maximum( sd_capslab, sd_infcyl )    ## CSG intersection 
    return sd

def sdf_pipe(x, y, z, z1=-5, z2=7, radius_inner=5, radius_outer=8 ):
    inner_sd = sdf_cylinder(x,y,z, z1=z1, z2=z2, radius=radius_inner )
    outer_sd = sdf_cylinder(x,y,z, z1=z1, z2=z2, radius=radius_outer )
    pipe_sd = np.maximum( outer_sd, -inner_sd )   ## CSG difference (intersection with complement)
    return pipe_sd

def sdf_pipe_box(x, y, z, z1=-5, z2=7, radius_inner=5, radius_outer=8, bx=1, by=2, bz=3  ):
    inner_sd = sdf_cylinder(x,y,z, z1=z1, z2=z2, radius=radius_inner )
    outer_sd = sdf_cylinder(x,y,z, z1=z1, z2=z2, radius=radius_outer )
    pipe_sd = np.maximum( outer_sd, -inner_sd )   # CSG subtraction 
    box_sd = sdf_box(x,y,z, bx=bx, by=by, bz=bz )

    pipe_box_sd = np.minimum( pipe_sd, box_sd )    # CSG union 
    return pipe_box_sd







n = 100
x_min, y_min, z_min = -10, -10, -10

dims=(n, n, n)
spacing=(abs(x_min)/n*2, abs(y_min)/n*2, abs(z_min)/n*2)
origin=(x_min, y_min, z_min)


grid = pv.UniformGrid(dims, spacing, origin)
print(grid)
x, y, z = grid.points.T


#values = sdf_sphere(x,y,z, radius=7)
#values = sdf_box(x,y,z)
#values = sdf_cylinder(x,y,z)
#values = sdf_pipe(x,y,z)
values = sdf_pipe_box(x,y,z, bx=9, by=2, bz=3 )

grid.point_arrays["values"] = values


method = ['marching_cubes','contour','flying_edges'][1]
isovalue = 0 
num_isosurfaces = 1 
mesh = grid.contour(num_isosurfaces, scalars="values", rng=[isovalue, isovalue], method=method )

SIZE = np.array([1280, 720])

pl = pv.Plotter(window_size=SIZE*2)
pl.add_mesh(mesh, smooth_shading=False, color='tan', show_edges=False  )   # style='wireframe' 

# split_sharp_edges=True  # probably this is only for future version

pl.show_grid()
pl.show()


