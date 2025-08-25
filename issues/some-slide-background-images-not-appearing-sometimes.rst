some-slide-background-images-not-appearing-sometimes
======================================================

Issue 
---------------

Some background images not showing on laptop.

Fix
----

Rerunning p-- made them show up. 

I think the problem must be that p-pull threw away local changes in order to pull in the 
new images and as a result lost the needed html changes. 
Updated simoncblyth.github.io reverting the HTML to 
an earlier version without those last background images configured.


Investigate
-------------

In Firefox Developer Edition, show source see::

    view-source:http://localhost/env/presentation/opticks_20250727_kaiping.html?page=0

And the URLs work:

* http://localhost/env/presentation/GEOM/J25_4_0_opticks_Debug/cxr_min/SPMT_cxs/20250722_155442.png
* http://localhost/env/presentation/GEOM/J25_4_0_opticks_Debug/SGLFW_SOPTIX_Scene_test/wp_pmt_semi_ipc/20250721_111242.png
* http://localhost/env/presentation/GEOM/J25_4_0_opticks_Debug/SGLFW_SOPTIX_Scene_test/wp_pmt_semi_ipc/20250721_111433.png
   



