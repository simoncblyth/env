# === func-gen- : matplotlib/mpl fgp matplotlib/mpl.bash fgn mpl fgh matplotlib
mpl-src(){      echo matplotlib/mpl.bash ; }
mpl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mpl-src)} ; }
mpl-vi(){       vi $(mpl-source) ; }
mpl-env(){      elocal- ; }
mpl-usage(){ cat << EOU

Matplotlib Notes
==================

See also matplotlib- ipython-


Refs
-----

* http://www.physics.ucdavis.edu/~dwittman/Matplotlib-examples/

ColorMaps
----------

* https://bids.github.io/colormap/

* http://colorspacious.readthedocs.org/en/latest/tutorial.html


imshow colors 
---------------

* http://stackoverflow.com/questions/24739769/matplotlib-imshow-plots-different-if-using-colormap-or-rgb-array


MPL/Python Color Tools
-----------------------

* http://colour-science.org/blog_a_plea_for_colour_analysis_tools_in_dcc_applications.php

* http://markkness.net/colorpy/ColorPy.html




Access photons hit data 4x4
-----------------------------

::

    p = ph("h1")
    gx,gy,gz,gt = p[:,0,0], p[:,0,1], p[:,0,2], p[:,0,3]  # global pos, time
    dx,dy,dz,dw = p[:,1,0], p[:,1,1], p[:,1,2], p[:,1,3]  # global direction, wavelength 
    px,py,pz,pw = p[:,2,0], p[:,2,1], p[:,2,2], p[:,2,3]  # global pol, weight
    pid,slot,flags,pmt = p[:,3,0].view(np.int32), p[:,3,1].view(np.int32), p[:,3,2].view(np.uint32), p[:,3,3].view(np.int32)


Access hit data 8x3 
---------------------

::

    a = hh("hh1")
    lx,ly,lz = a[:,3,0], a[:,3,1], a[:,3,2]



Version
--------

::

    In [18]: matplotlib.__version__
    Out[18]: '1.3.1'


3D scatter plot
----------------

* http://matplotlib.org/1.3.1/mpl_toolkits/mplot3d/tutorial.html


Checking global hit positions look reasonable::

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( px,py,pz )
    plt.show()


    ax.scatter( lx,ly,lz )



2D Quiver
------------

Shows hit directions are outwards::

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver( gx,gy, dx,dy  )
    plt.show()


3D Quiver plot
----------------

Need 1.4 

::

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver( gx,gy,gz, dx,dy,dz  )
    plt.show()





EOU
}
mpl-dir(){ echo $(env-home)/matplotlib ; }
mpl-cd(){  cd $(mpl-dir); }
mpl-mate(){ mate $(mpl-dir) ; }
mpl-get(){
   local dir=$(dirname $(mpl-dir)) &&  mkdir -p $dir && cd $dir

}
