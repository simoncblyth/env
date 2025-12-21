# === func-gen- : presentation/presentation fgp presentation/presentation.bash fgn presentation fgh presentation
presentation-src(){      echo presentation/presentation.bash ; }
presentation-source(){   echo ${BASH_SOURCE:-$(env-home)/$(presentation-src)} ; }
presentation-vi(){       vi $(presentation-source) ; }
presentation-env(){      elocal- ; }


presentation-usage(){ cat << EOU

Presentation preparation
============================

See Also
-----------

* slides-;slides-vi


Upcoming presentation : Hohhot
-------------------------------

Tentative title::

    Standalone C++ software testing, debugging and analysis with python packages:  NumPy, matplotlib, pyvista, ...


FUNCTIONS
---------

presentation-index
    open local index.html in browser 

HTML Character Codes for inclusion into RST of presentations
--------------------------------------------------------------

* https://www.rapidtables.com/web/html/html-codes.html

* less-than greater-than and hash are problematic within html/rst so use HTML character codes:

  * less than &lt;
  * greater than &gt;
  * hash &#35;  

Debug no show image : turns out to be due to case sensitivity
-------------------------------------------------------------------

Get URL from safari console as a failed to load resource

* https://simoncblyth.bitbucket.io/env/presentation/CSG/tests/CSGTargetGlobalTest/solidXJfixture:64_radii.png
* /env/presentation/CSG/tests/CSGTargetGlobalTest/solidXJfixture:64_radii.png

Try renaming::

    epsilon:CSGTargetGlobalTest blyth$ git mv solidXJfixture:64_radii.png solidXJfixture64radii.png
    epsilon:CSGTargetGlobalTest blyth$ git s
    On branch master
    Your branch is up-to-date with 'origin/master'.

    Changes to be committed:
      (use "git reset HEAD <file>..." to unstage)

        renamed:    solidXJfixture:64_radii.png -> solidXJfixture64radii.png

    epsilon:CSGTargetGlobalTest blyth$ 


Looking at the repository source reveals the reason for noshow. 
It is because of a clash between "CSG" and "csg" directories. 

https://bitbucket.org/simoncblyth/simoncblyth.bitbucket.io/src/master/env/presentation/csg/tests/CSGTargetGlobalTest/solidXJfixture64radii.png

epsilon:presentation blyth$ mkdir CSGTargetGlobalTest
epsilon:presentation blyth$ git mv CSG/tests/CSGTargetGlobalTest/solidXJfixture64radii.png CSGTargetGlobalTest/
fatal: not under version control, source=env/presentation/CSG/tests/CSGTargetGlobalTest/solidXJfixture64radii.png, destination=env/presentation/CSGTargetGlobalTest/solidXJfixture64radii.png
epsilon:presentation blyth$ 
epsilon:presentation blyth$ 
epsilon:presentation blyth$ git mv csg/tests/CSGTargetGlobalTest/solidXJfixture64radii.png CSGTargetGlobalTest/


PP warning
------------

#pragma GCC diagnostic ignored "-Winvalid-pp-token"


Creating slides PDF with talk annotations interleaved
-------------------------------------------------------

See instructions in slides-;slides-vi


macOS Apache Directory : /Library/WebServer/Documents
--------------------------------------------------------

Local slide presentation using apache: 

* http://localhost/env/presentation/juno_opticks_20210712.html

is arranged via the "env" symbolic link in /Library/WebServer/Documents 

::

    epsilon:Documents blyth$ pwd
    /Library/WebServer/Documents
    epsilon:Documents blyth$ l env
    0 lrwxr-xr-x  1 root  wheel  41 Jun 25  2018 env -> /Users/blyth/simoncblyth.bitbucket.io/env


to use the same resources as remote bitbucket presentation from 
bitbucker servers uses, 

* https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210712.html

once any changes are pushed there.

::

    epsilon:my-small-white blyth$ pwd
    /Library/WebServer/Documents/env/presentation/ui/my-small-white
    epsilon:my-small-white blyth$ realpath $PWD
    /Users/blyth/simoncblyth.bitbucket.io/env/presentation/ui/my-small-white
    epsilon:my-small-white blyth$ 



How it works : rst2s5-2.6.py and ".. s5_background_image::"
--------------------------------------------------------------

The "presentation--" function runs a Makefile 
from (presenstation-cd) which invokes ./rst2s5-2.6.py

The list of images and layout instructions from the s5_background_image
directive in presentations is read by s5_background_image.py 
Yielding: url, size, position, extra

These feed directly into the html/css directives of a div element:: 

   background-image: url(%(url)s);        
   background-size: %(size)s;
   background-position: %(position)s;
   %(extra)s

* https://developer.mozilla.org/en-US/docs/Web/CSS/background-size
* https://developer.mozilla.org/en-US/docs/Web/CSS/background-position

Underscores in the rst directive are replaced with spaces.::

    G4OpticksTest_june2020_p1
    /env/presentation/fermilab_geant4_team/G4OpticksTest_june2020_p1_half.png 1180px_620px 0px_0px

Observations as change background-size and background-position::

   1280px_720px 0px_0px : fills standard sized page   
   1180px_620px 0px_0px : shrinks with top left staying as is, leaving gap to bottom right
   1180px_620px 50px_50px : offsets by 50px horizontally and vertically, centering 100px shrunk the slide with gaps all around 


Local Rough Index of All Presentations
-----------------------------------------

Shortcut::

    presentation-;presentation-index

    # does the below 
    open http://localhost/env/presentation/index.html

    # to check without web server
    open file://$HOME/simoncblyth.bitbucket.io/env/presentation/index.html


Updating the rough index of all presentations
-----------------------------------------------

::

    epsilon:presentation blyth$ cat /Users/blyth/env/bin/index.sh
    #!/bin/bash 

    index.py 
    open http://localhost/env/presentation/index.html

::

    epsilon:presentation blyth$ which index.py 
    /Users/blyth/env/bin/index.py


Public Index of Selected Presentations
-----------------------------------------

* https://simoncblyth.bitbucket.io


To update the public index
----------------------------

::

   # add an entry for the presentation to the RST 
   cd ~/env/simoncblyth.bitbucket.io/ && vi index.txt && make  

   # local preview the new index in Safari 
   open /Users/blyth/simoncblyth.bitbucket.io/index.html 

   # push the HTML index and presentation to bitbucket 
   cd ~/simoncblyth.bitbucket.io

   git status 
   git pull
   git add ...
   git commit -m "add HSF presentation and index entry"
   git push 

   open http://simoncblyth.bitbucket.io


* caution that ~/simoncblyth.bitbucket.io not really a source repo, as it contains
  lots of html derived from RST as well as binary PNG screenshots, plots etc..


Sizes
------

::

    1280x720

    1280x1440   double up the height 

    In [6]: 1440./(1280./210.)
    Out[6]: 236.25

    A4: 210x297

    ->  210x237

    # Preview Paper Size is whacky when trying to get 2-up layout 
    # to avoid blank space : by expt 100x140 works OK


Convert HTML pages into PDF
------------------------------

0. exit Safari and reopen 
1. presentation-- : make and display 
2. slides-safari : adjust Safari screen size
3. make sure Safari font size is normal with images fitting pages
4. slides-get 0 2 

::

   slides--
   slides--t  


Jump to a slide page 
----------------------

* **To jump to a slide : type 0-based slide number and press return** 

  * this works both for http://localhost as well as remote  
  * TODO: check file::// urls


Notes on use of the html
--------------------------

The PDF is bitmapped. For copy/paste functionality use the html or RST source or available at:

https://simoncblyth.bitbucket.io/env/presentation/opticks_gpu_optical_photon_simulation_oct2018_ihep.html
https://simoncblyth.bitbucket.io/env/presentation/opticks_gpu_optical_photon_simulation_oct2018_ihep.txt

Once the images complete loading you can navigate the html “slides” with
javascript selector at bottom right, or arrow keys, or jump to a page by entering 
the page number and pressing return in your browser.  

JS changes causing problems
-----------------------------

::

    epsilon:my-small-white blyth$ hg log slides.js
    changeset:   6443:0e66b24c9140
    tag:         tip
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Thu Dec 12 21:38:38 2019 +0800
    summary:     adding introductory slides : neutrino, IBD, scintillation, cerenkov, PMT

    changeset:   6438:e553471f46c1
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Fri Dec 06 20:27:15 2019 +0800
    summary:     make it possible to hide the currentSlide and navLinks in the s5 js slides, using a and b keys

    changeset:   4681:bc8801f6adab
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Tue Sep 23 20:11:21 2014 +0800
    summary:     moving presentation to reflect generality

    epsilon:my-small-white blyth$ hg revert -r 4681:bc8801f6adab slides.js 



Safari Caching of Javascript drives bonkers
---------------------------------------------

* recent experience with Safari much smoother, success with just : Develop > Empty Caches 
* previously had trouble getting updates to be honoured
* https://stackoverflow.com/questions/43462424/reload-javascript-and-css-files-for-a-website-in-safari

::

    <filesMatch "\.(html|htm|js|css)$">
      FileETag None
      <ifModule mod_headers.c>
         Header unset ETag
         Header set Cache-Control "max-age=0, no-cache, no-store, must-revalidate"
         Header set Pragma "no-cache"
         Header set Expires "Wed, 21 Oct 2015 07:28:00 GMT"
      </ifModule>
    </filesMatch>


Keystroke to safari ?
------------------------

::

   osascript -e 'tell application "System Events" to key code 0 '


For Conversion of some slides to Keynote : need to switch off page numbers and RHS interface gumpf 
-----------------------------------------------------------------------------------------------------------

Safari > Develop > Start Element Selection (shift-cmd-C) 
      allows to click on elemnt of page and see the source 

::

   <div id="currentSlide" style="visibility: visible;"> 
      # contains some spans with the the 2/2 or whatever

   document.getElementById("currentSlide").style.visibility = "hidden"
   document.getElementById("currentSlide").style.visibility = "visible"

   <div id="navLinks" > 
      # contains the toggle and prev/next buttons 

   document.getElementById("navLinks").style.visibility = "hidden"
   document.getElementById("navLinks").style.visibility = "visible"


Huh, seems Makefile doesnt copy the UI javascript into deployed
location::

    epsilon:ui blyth$ cp -r my-small-white ~/simoncblyth.bitbucket.io/env/presentation/ui/
    epsilon:ui blyth$ pwd
    /Users/blyth/env/presentation/ui



Getting rid of the presentation title in the footer on some slides ?
-----------------------------------------------------------------------

::

   document.getElementById("footer").getElementsByTagName("h1")[0].style.visibility = "invisible"


* http://localhost/env/presentation/ui/my-small-white/pretty.css

~/env/presentation/ui/my-small-white/pretty.css::

     22 div#footer {font-family: sans-serif; color: #444;
     23   font-size: 0.5em; font-weight: bold; padding: 1em 0; visibility: hidden;}



::

    epsilon:my-small-white blyth$ ./update.sh 
    cp /Users/blyth/env/presentation/ui/my-small-white/pretty.css /Users/blyth/simoncblyth.bitbucket.io/env/presentation/ui/my-small-white/pretty.css
    15c15,16
    < var snumdiv = 1;
    ---
    > var talk = window.location.href.indexOf('_TALK') > -1 ? 1 : 0 ;
    > var snumdiv = talk == 1 ? 2 : 1;
    24a26,27
    > 
    > 
    128c131
    < 	cs.innerHTML = '<span id="csHere">' + snum/snumdiv + '<\/span> ' + 
    ---
    > 	cs.innerHTML = '<span id="csHere">' + Math.floor(snum/snumdiv) + '<\/span> ' + 
    130,131c133,134
    < 		           '<span id="csTotal">' + (smax-1)/snumdiv + '<\/span>' ;
    <     if (snum > -1) {
    ---
    > 		           '<span id="csTotal">' + Math.floor((smax-1)/snumdiv) + '<\/span>' ;
    > 	if (snum == 0) {
    cp /Users/blyth/env/presentation/ui/my-small-white/slides.js /Users/blyth/simoncblyth.bitbucket.io/env/presentation/ui/my-small-white/slides.js
    epsilon:my-small-white blyth$ 



Opticks Oct 2018 JUNO Detector Video 
---------------------------------------

* https://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=3927


Preprocessor phase for single source yielding multiple versions of presentation
---------------------------------------------------------------------------------

Note that the preprocessor leaves blocks of empty lines where the
"#ifdef" and non-picked blocks were. So it is necessary to be cautions 
precisely where to place the "#ifdef" "#else" "#endif"  as the 
the resulting blocks of blank lines can be significant to RST.

* http://jkorpela.fi/html/cpre.html

::

   gcc -E -P -x c -traditional-cpp   -DTOKEN  t.rst    # emits to stdout with ifdef etc... obeyed 

E
   use preprocessor only
P
   dont add line directives   
C
   smth to do with comments 
traditional-cpp
   avoid trimming trailing whitespace touching and collapsing multi-char whitespace to one char, 
   which breaks RST gridtables 

   * :google:`prevent C preprocessor from collapsing whitespace`
   * https://stackoverflow.com/questions/445986/how-to-force-gcc-preprocessor-to-preserve-whitespace


* NB switch all # comments in RST to use ## to avoid invalid preprocessor directive errors


Workflow for preparing slides
------------------------------

1. in one Terminal tab edit presentation txt with presentation-edit
2. in another, convert the rst to html and open current page in Safari with:: 

   PAGE=20 presentation--

* NB **DO NOT** simply reload and page forwards, instead use the above to jump to the page, 
  can also "presentation-open 10" or "po 10" to jump to a page without rebuilding the html


NVIDIA Turing Press deck 
---------------------------

* https://www.anandtech.com/Gallery/Album/6660#27
* https://www.irisa.fr/alf/downloads/collange/talks/ufmg_scollange.pdf

  Architecture and micro-architecture of GPUs

  SIMD is static, SIMT is dynamic


S5 : Presentation HTML mechanics : jumping to a slide during presentation preparation
---------------------------------------------------------------------------------------

* http://docutils.sourceforge.net/docs/user/slide-shows.html

* ui/my-small-white/slides.js is javascript in use 

* agressive caching of javascript by Safari seems undefeatable 

  * force a fail to load by changing the name of ui/my-small-white/slides.js to ensure a change gets seen  

* **To jump to a slide : type the number and press return** 

* http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_sep2018_qingdao.html?p=30

  * Changing ?p=xx of url in browser and reloading, does not work in Safari, it does in Chrome 
    
    * actually it kinda works : initially goes to first page, a subsequent reload goes to desired page 
    * BUT: can just type page number whilst page is in focus and press return in both Safari/Chrome

  * open from commandline with "?p=xx" url creates a new Safari tab and loads correctly  

    * "open http://localhost/env/presentation/opticks_gpu_optical_photon_simulation_sep2018_qingdao.html?p=30"
    * "presentation-open 30"   
    * use shortcut function for this : po () { presentation-; presentation-open ${1:-0} ; }
    * get into habit of closing tabs with cmd-W, as using po creates new ones all the time

     
Shrinking PDFs with ColorSync/Colorsync Utility
------------------------------------------------

Usage Steps
~~~~~~~~~~~~

1. spotlight search for "ColorSync Utility.app"
2. select "Filters" tab
3. find the "Reduce File Size Copy" filter, open the 
   disclosure to see the JPEG compression factor that will be applied
4. use "File > Open" in "ColorSync Utility" to open the uncompressed PDF
5. pick the "Reduce File Size Copy" using drop down menu at bottom left
6. press the "Apply" button at bottom right
7. use "File > Save As" to save the result to a different name
   (eg nacent convention of mine is to append an undercore before ".pdf")

   * actually trailing underscores has special RST meaning hence it is 
     better to just be explicit and add suffix "_compressed.pdf"

Compare sizes::

    epsilon:Desktop blyth$ du -hs opticks_gpu_optical_photon_simulation_jul2018_chep*
     15M	opticks_gpu_optical_photon_simulation_jul2018_chep.pdf
     49M	opticks_gpu_optical_photon_simulation_jul2018_chep_uncompressed.pdf 

    epsilon:Desktop blyth$ du -h opticks_nov2019*
     74M	opticks_nov2019_chep.pdf
     20M	opticks_nov2019_chep_.pdf

    epsilon:Desktop blyth$ du -h *hsf*
     51M	opticks_may2020_hsf.pdf
     18M	opticks_may2020_hsf_.pdf
     62M	opticks_may2020_hsf_TALK.pdf


Setup Steps
~~~~~~~~~~~~~~~

* https://www.cultofmac.com/481247/shrink-pdfs-colorsync-utility/

* use "Live update from filter inspector"
* copied the existing "Reduce File Size" filter
* modified "Reduce File Size Copy" removed Image Sampling, set Image Compression to JPEG 75% 
  
  * PDF size reduced from 49M to 15M and quality reduction difficult to notice

::

    epsilon:Desktop blyth$ du -hs opticks*
     49M	opticks_gpu_optical_photon_simulation_jul2018_chep.pdf
     15M	opticks_gpu_optical_photon_simulation_jul2018_chep_j75.pdf


TODO
-----

* DONE : enable issues on bitbucket
* prepare a list of email addresses of the Opticks interested
* write Opticks News

  * whats new : CMake 3.5+, BCM, target export -> easy config 
  * WIP : direct from G4 geometry
  * issues enabled on bitbucket
  * mailing list : ground rules, not an issue tracker : use bitbucket for that

* send Opticks News to the interested list, invite to mailing list, inform on issue tracker

* https://groups.io/g/opticks/promote

  * embed signup form into bitbucket 


Machinery Fixes
----------------

* Makefile had some old simoncblyth.bitbucket.org rather than simoncblyth.bitbucket.io
* apache changes in High Sierra, see hapache-

Potential Customers
-----------------------

::

    SNO+     : large scale liquid scintillator expt 
    WATCHMAN : 
    THEIA
    CheSS   


Sep 2017 Wollongong 
---------------------

22nd Geant4 Collaboration Meeting, UOW Campus, Wollongong (Australia), 25-29 September 2017.

intro
~~~~~~

The main detector consists of a 35.4 m (116 ft) diameter transparent acrylic
glass sphere containing 20,000 tonnes of linear alkylbenzene liquid
scintillator, surrounded by a stainless steel truss supporting approximately
53,000 photomultiplier tubes (17,000 large 20-inch (51 cm) diameter tubes, and
36,000 3-inch (7.6 cm) tubes filling in the gaps between them), immersed in a
water pool instrumented with 2000 additional photomultiplier tubes as a muon
veto.[8]:9 Deploying this 700 m (2,300 ft) underground will detect neutrinos
with excellent energy resolution.[3] The overburden includes 270 m of granite
mountain, which will reduce cosmic muon background.[9]

The much larger distance to the reactors (compared to less than 2 km for the
Daya Bay far detector) makes the experiment better able to distinguish neutrino
oscillations, but requires a much larger, and better-shielded, detector to
detect a sufficient number of reactor neutrinos.

renders to make
~~~~~~~~~~~~~~~~~~

* j1707 InstLODCull 
* j1707 analytic



Dear Visualisation, Geometry and EM Coordinators,..

I think my ongoing work on accelerating optical photon simulation
using the NVIDIA OptiX GPU ray tracer in the context of PMT based
neutrino detectors such as JUNO or Daya Bay may be of interest 
to your subgroups. 

I presented my work at the 2014 Okinawa Collaboration Meeting.
My work has matured greatly since then, if I were to receive an
invitation to attend the upcoming Geant4 Collaboration Meeting in Australia
I may be able to secure funding to attend, enabling me to present/discuss
my work with everyone interested in GPU accelerated optical photon 
simulation and GPU visualization. 

My work is available in the open source Opticks project

* https://bitbucket.org/simoncblyth/opticks/

Numerous status reports, conference presentations and videos 
are linked from https://simoncblyth.bitbucket.io including the 
latest focusing on CSG on GPU.

* https://simoncblyth.bitbucket.io/env/presentation/opticks_gpu_optical_photon_simulation_jul2017_ihep.html
 (it takes a several seconds for the slide presentation to load)

Over the past 6 months I have succeeded to implement general 
CSG binary tree intersection on the GPU within an NVIDIA OptiX 
intersect "shader" and have used this capability to develop 
automatic translation of GDML geometries into a fully 
analytic GPU appropriate forms, using the glTF 3D file format 
as an intermediate format.
Entire detector geometries including all material and optical surface properties
are translated, serialized into buffers and copied to the GPU 
at initialization. 

This allows full analytic geometries, without triangulation approximation, 
to be auto-translated and copied to the GPU meaning that in principal 
GPU ray intersections should very closely match those obtained with Geant4, 
as the GPU and CPU are doing the same thing, finding roots of  
polynomials with the same coefficients.  OpenGL/OptiX GPU instancing 
techniques are used to efficiently handle geometries with ~30k PMTs and 
BVH (boundary volume heirarchy) structure acceleration comes
for free with the NVIDIA OptiX ray tracing framework.

GPU CSG geometry together with CUDA/OptiX ports of optical photon 
physics allows optical photons to be simulated entirely on the GPU. 
Photon generation (from G4Cerenkov, G4Scintillation) 
and propagation (G4OpAbsorption, G4OpBoundaryProcess, G4OpRayleigh)
have been ported.
Of the many millions of optical photons simulated per event
only the small fraction that hit photon detectors need to be copied 
to the CPU where they can populate the standard Geant4 hit collections.
Integration with Geant4 is currently done via modified G4Scintillation
and G4Cerenkov processes which collect "gensteps" that are copied to the GPU.

Simulations limited by optical photons can benefit hugely from 
the GPU parallelism that Opticks makes accessible with optical
photon speedup factors estimated to exceed 1000x for the JUNO 
experiment.

Opticks needs to mature through production usage within the JUNO
experiment(and perhaps others) before it makes sense to consider details
of any "formal" integration, nevertheless looking ahead
I am interested to learn opinions of Geant4 members as to whether
that direction is even feasible ? And what form it might take if it were.

The bottom line is that I think a significant number of experiments
can benefit greatly from using Opticks together with Geant4 and I would
like to help them do so.

Simon C. Blyth,  National Taiwan University


update CHEP talk : ie intro to someone never having seen Opticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* review progress since CHEP (~1 year) doing appropriate updates, analytic CSG  

Dear Laurent, 

Thanks.   I’ve recently made some progress that I guess will be particularly interesting 
to you and the Visualisation group,  on improving OpenGL performance for very large geometries,
specifically the JUNO Neutrino detector: 

   18,000 20inch PMTs of ~5000 triangles each, 
    36,000 3inch PMTs of ~1500 triangles each,
   nominal total of ~150M triangles

Using dynamic instance culling and variable level-of-detail meshes for instances ( the PMTs) 
based on distance to the instance.  These use GPU compute 
(OpenGL transform feedback streams with geometry shader, so not NVIDIA specific) 
prior to rendering each frame in order to skip instances that are not visible and replace 
distant instances with simpler geometry.   

On my Macbook Pro 2013 laptop with NVIDIA GT 750M the optimisation turns a formerly painful ~2fps
experience into  >30fps  of comfortable interactivity. Plus the performance can be tuned for the available
GPU by adjusting the distances at which to switch on different levels of detail.  

Its a rather neat technique as the optimisation is cleanly split from the actual rendering,
which remains unchanged other that getting its 4x4 instance transforms from the dynamic buffers
for each level of detail.

I’ll hope we’ll find you a time slot to present your work.

Me too!  How long a presentation would you like ?


Simon

Workflow
-----------

Preparation workflow:

#. change presention name below and create the .txt
#. iterate on presentation by running the below which invokes *presentation-make*
   and *presentation-open* to view local html pages in Safari::

   presentation.sh

#. static .png etc.. are managed within bitbucket static repo, 
   local clone at ~/simoncblyth.bitbucket.org

   * HMM NOW ~/simoncblyth.bitbucket.io 

   * remember not too big, there is 1GB total repo limit 

#. running presentation.sh updates the derived html within
   the static repo clone, will need to "hg add" to begin with


Publishing to remote:

#. update index page as instructed in *bitbucketstatic-vi*
#. push the statics to remote 


#. creating PDFs, see slides-;slides-vi add eg "slides-get-jul2017" 
   depending on slide count


Creating retina screencaptures
---------------------------------

* shift-cmd-4 and drag out a marquee, this writes .png file to Desktop

::

   cd ~/env/presentation   ## cd to appropriate directory for the capture

   osx_                    ## precursor define the functions
   osx_ss-cp name          

   ## copy last screencapture from Desktop to corresponding relative dir beneath ~/simoncblyth.bitbucket.org 
   ## this is the local clone of the bitbucket statics repo

   cd ~/simoncblyth.bitbucket.io/env/presentation/whereever/ 
   osx_ss_copy name    # no directory cleverness, just copies into pwd and downsizes the screencapture yielding name_half.png 


Incorporating retina screencaptures
-------------------------------------

::

    simon:presentation blyth$ cd ~/simoncblyth.bitbucket.org/
    simon:simoncblyth.bitbucket.org blyth$ downsize.py env/graphics/ggeoview/PmtInBox-approach.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize env/graphics/ggeoview/PmtInBox-approach.png to create env/graphics/ggeoview/PmtInBox-approach_half.png 2138px_1538px -> 1069px_769px 
    simon:simoncblyth.bitbucket.org blyth$ 

s5 rst underpinning
--------------------

* http://docutils.sourceforge.net/docs/user/slide-shows.html#s5-theme-files


Presentation Structure
------------------------

* https://hbr.org/2012/10/structure-your-presentation-li

* What-Is/What-could-be 


Opticks Overview : Narrative Arc
-----------------------------------


* optical photon problem of neutrino detectors

* NVIDIA OptiX

Let me pick up the story from ~1yr ago Oct 2016 (CHEP) 

* approx 1yr ago, Opticks was mostly using tesselated approximate geometries
  with some manual PMT analytic conversions

* after lots of work : validation chi-sq comparisons were showing an excellent 
  match between the GPU ported simulation and standard G4,
  BUT with a great big CAVEAT : **purely analytic geometry**

  * so i had got the optical physics to the GPU  
  * BUT: only approximate tesselated geometry, with some manual analytic  


Requirements were clear:

* intersection code for rays with CSG trees 
* auto-conversion of GDML into OpticksCSG for upload to GPU 


Additions
------------

* Large Geometry Techniques : instance culling, LOD
* Primitives : Torus, Ellipsoid
* CSG non-primitives, complement 

 

presentations review, highlighting new developments
------------------------------------------------------

2017
~~~~~


opticks_gpu_optical_photon_simulation_sep2017_wollongong.txt
     update oct2016_chep with 1 year of progress (mainly analytic CSG)

opticks_gpu_optical_photon_simulation_jul2017_ihep.txt
     mostly on CSG

opticks_gpu_optical_photon_simulation_jan2017_psroc.txt
     same as LLR 

2016
~~~~~~

opticks_gpu_optical_photon_simulation_nov2016_llr.txt
    focus on optical photon validation comparisons in simple analytic geomtry

    * tconcentric : message "match achieved after many fixes" 
    * excellent match, loath to loose that with approximate geometry -> analytic implementation


opticks_gpu_optical_photon_simulation_oct2016_chep.txt
    validation start, chisq minimization
    
    * ioproc-open


opticks_gpu_optical_photon_simulation_jul2016_weihai.txt

opticks_gpu_optical_photon_simulation_may2016_lecospa.txt

opticks_gpu_optical_photon_simulation_april2016_gtc.txt

opticks_gpu_optical_photon_simulation_march2016.txt

optical_photon_simulation_progress.txt
     Opticks : GPU Optical Photon Simulation using NVIDIA OptiX
     Jan 2016

opticks_gpu_optical_photon_simulation_psroc.txt
     Jan 2016 

opticks_gpu_optical_photon_simulation.txt
     Jan 2016 



2015
~~~~~
    
optical_photon_simulation_with_nvidia_optix.txt
     Optical Photon Simulation with NVIDIA OptiX
     July 2015

gpu_optical_photon_simulation.txt
     200x Faster Optical Photon Propagation with NuWa + Chroma ?
     Jan 2015

gpu_accelerated_geant4_simulation.txt
     GPU Accelerated Geant4 Simulation with G4DAE and Chroma
     Jan 2015 
 
2014
~~~~~

g4dae_geometry_exporter.txt
     G4DAE : Export Geant4 Geometry to COLLADA/DAE XML files
     19th Geant4 Collaboration Meeting, Okinawa, Sept 2014


g4dae_geometry_exporter (Sept 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**G4DAE : Export Geant4 Geometry to COLLADA/DAE XML files**

9th Geant4 Collaboration Meeting, Okinawa, Sept 2014

Why export into DAE ?

Ubiquitous geometry visualization for Geant4 users and outreach. 
Facilitate innovative use of geometry data.

Moving geometry to GPU and implementing simple shaders
unleases performant visualization 

* Exporter details
* What is COLLADA/DAE ?
* Validating exports : compare with GDML and VRML2
* General viewing of exports
* Custom use : bridging to GPU
* OpenGL Viewer implementation
* Optical Photon Data handling
* Introducing Chroma
* Chroma raycasting
* Chroma photon propagation
* G4DAE exporter status


gpu_accelerated_geant4_simulation (Jan 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Accelerated Geant4 Simulation with G4DAE and Chroma** 


* Geometry Model Implications
* Geant4 <-> Chroma integration : External photon simulation workflow
* G4DAE Geometry Exporter
* Validating GPU Geometry
* G4DAEChroma bridge
* Chroma forked
* Validating Chroma generated photons
* Next Steps
* Visualizations 

Track > Geometry intersection typically limits simulation performance. 
Geometry model determines techniques (and hardware) available to accelerate
intersection.

*Geant4 geometry model (solid base)*
    Tree of nested solids composed of materials, each shape represented by different C++ class

*Chroma Geometry model (surface based)*
    List of oriented triangles, each representing boundary between inside and outside materials.

3D industry focusses on surface models >> frameworks 
and GPU hardware designed to work with surface based geometries.


*Geometry Set Free*

Liberating geometry data from Geant4/ROOT gives free choice of visualization
packages. Many commercial, open source apps/libs provide high performance
visualization of DAE files using GPU efficient OpenGL techniques. 
Shockingly Smooth Visualization performance

**Above not really true: better to say surface based geometry is a better fit for ray tracing**

BUT Chroma needs : triangles + inside/outside materials

Chroma tracks photons through a triangle-mesh detector geometry, simulating
processes like diffuse and specular reflections, refraction, Rayleigh
scattering and absorption. Using triangle meshes eliminate geometry code as
just one code path.

Optical photons (modulo reemission) are the leaves of the simulation tree,
allowing external simulation to be integrated rather simply.

* generation of Cerenkov and Scintillation Photons based on Geant4 Generation Step inputs


gpu_optical_photon_simulation (Jan 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**200x Faster Optical Photon Propagation with NuWa + Chroma ?**

Fivefold path:

* Export G4 Geometry as DAE
* Convert DAE to Chroma
* Chroma Validation, raycasting
* Chroma Stability/Efficiency Improvements Made
* G4/Chroma Integration
* Chroma vs G4 Validation 


2015-01-06 : last commit to Chroma fork
2015-01-20 : first commit to Opticks "try out NVIDIA Optix 301" https://bitbucket.org/simoncblyth/opticks/commits/bd1c43

 
optical_photon_simulation_with_nvidia_optix (July 2015}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Optical Photon Simulation with NVIDIA OptiX**

*OptiX : performance scales with CUDA cores across multiple GPUs*

* Why not Chroma ?
* Introducing NVIDIA OptiX

  * Parallels between Realistic Image Synthesis and Optical Simulation
  * OptiX Programming Model

* OptiX testing

  * OptiX raycast performance
  * OptiX Performance Scaling with GPU cores

* New Packages Replacing Chroma

  * Porting Optical Physics from Geant4/Chroma into OptiX
  * Optical Physics Implementation
  * Random Number Generation in OptiX programs (initialization stack workaround)
  * Fast material/surface property lookup from boundary texture
  * Reemission wavelength lookup from Inverted CDF texture
  * Recording the steps of ~3 million photons 
  * Scintillation Photons colored by material
  * Indexing photon flag/material sequences
  * Selection by flag sequence

* Mobile GPU Timings
* Operation with JUNO Geometry ?
* Next Steps


opticks_gpu_optical_photon_simulation_psroc (Jan 2016)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation**

**Opticks integrates Geant4 simulations with state-of-the-art NVIDIA OptiX GPU ray tracing**

* DayaBay, JUNO can expect: Opticks > 1000x G4 (workstation GPUs)


* Neutrino Detection via Optical Photons
* Optical Photon Simulation Problem
* NVIDIA OptiX GPU Ray Tracing Framework
* Brief History of GPU Optical Photon Simulation Development
* Introducing Opticks : Recreating G4 Context on GPU

* Validating Opticks against Theory

  * Opticks Absolute Reflection compared to Fresnel expectation
  * Opticks Prism Deviation vs Incident angles for 10 wavelengths
  * Multiple Rainbows from a Single Drop of Water

* Validating Opticks against Geant4

  * Disc beam 1M Photons incident on Water Sphere (S-Pol)
  * 2ns later, Several Bows Apparent
  * Rainbow deviation angles
  * Rainbow Spectrum for 1st six bows
  * 1M Rainbow S-Polarized, Comparison Opticks/Geant4

* Opticks Overview

  * Performance Comparison Opticks/Geant4

Three levels of geometry:

* OptiX: analytic intersection code
* OpenGL: tesselation for visualization
* Geant4: geometry construction code in CfG4 package


opticks_gpu_optical_photon_simulation (Jan 2016, Xiamen)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation**

Executive Summary

* Validation comparisons with Geant4 started, rainbow geometry validated
* Performance factors so large they become irrelevant (>> 200x)

* Brief History of GPU Optical Photon Simulation Development
* Introducing Opticks : Recreating G4 Context on GPU
* Large Geometry/Event Techniques
* Mesh Fixing with OpenMesh surgery

  * G4Polyhedron Tesselation Bug
  * OpenMeshRap finds/fixes cleaved meshes

* Analytic PMT geometry description

  * Analytic PMT geometry : more realistic, faster, less memory
  * Analytic PMT in 12 parts instead of 2928 triangles
  * OptiX Ray Traced Analytic PMT geometry
  * Analytic PMTs together with triangulated geometry
 
* Opticks/Geant4 : Dynamic Test Geometry

  * Opticks Absolute Reflection compared to Fresnel expectation

* Rainbow Geometry Testing

  * Opticks/Geant4 Photon Step Sequence Comparison


Optical Photons now fully GPU resident
All photon operations now done on GPU:

* seeded (assigned gensteps)
* generated
* propagated
* indexed material/interaction histories



opticks_gpu_optical_photon_simulation_march2016
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation**


* Validation comparisons with Geant4 advancing, single PMT geometry validated


* PmtInBox test geometry
* Single PMT Geometry Testing
* PmtInBox at 1.8ns
* PMT Opticks/Geant4 Step Sequence Comparison

  * Good agreement reached, after several fixes: geometry, TIR, GROUPVEL
  * nearly identical geometries (no triangulation error)

  * PMT Opticks/Geant4 step comparison TO BT [SD] : position(xyz), time(t)
  * PMT Opticks/Geant4 step comparison TO BT [SD] : polarization(abc), radius(r)
  * PmtInBox Opticks/Geant4 Chi2/ndf distribution comparisons
  * PmtInBox issues : velocity of photon propagation

* Photon Propagation Times Geant4 cf Opticks
* External Photon Simulation Workflow


opticks_gpu_optical_photon_simulation_april2016_gtc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Opticks : Optical Photon Simulation for Particle Physics with NVIDIA® OptiX™**

Nothing much new in this one.


opticks_gpu_optical_photon_simulation_may2016_lecospa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing much new in this one. Lots of intro.


opticks_gpu_optical_photon_simulation_oct2016_chep (20 pages, ~15 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


opticks_gpu_optical_photon_simulation_nov2016_llr  (32 pages, ~30 min)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Opticks Event Buffers  (technicality)
* Time/Memory profile of multi-event production mode running (~technicality)

* tconcentric : test geometry configured via boundaries
* tconcentric : fixed polarization "Torch" light source
* tconcentric : spherical GdLS/LS/MineralOil
* tconcentric : Opticks/Geant4 chi2 comparison
* tconcentric : Opticks/Geant4 history counts chi2/df ~ 1.0

* Group Velocity problems -> Time shifts  (~technicality)





EOU
}
presentation-dir(){ echo $(env-home)/presentation ; }
presentation-bdir(){ echo $HOME/simoncblyth.bitbucket.io/env/presentation ; }
presentation-index-(){ index.py $(presentation-bdir) ; }
presentation-index(){ open -a Safari.app http://localhost/env/presentation/index.html ; }
p-index(){  presentation-index ; }


presentation-c(){   cd $(presentation-dir); }
presentation-cd(){  cd $(presentation-dir); }

presentation-b(){   cd $(presentation-bdir); }
presentation-bcd(){  cd $(presentation-bdir); }


presentation-ls(){   presentation-cd ; ls -1t *.txt ; }
presentation-txts(){ presentation-cd ; vi $(presentation-ls) ;  }


presentation-notes(){ cat << EON

To render the html as PDF use::

   slides-get--
   slides-get-talk--


EON
}


collab-e(){

    presentation-cd
    vi \
$(presentation-iname).txt \
s5_background_image.txt \
juno_opticks_cerenkov_20210902.txt \
opticks_autumn_20211019.txt \
opticks_20211117.txt \
opticks_20211223_pre_xmas.txt  \
opticks_20220115_innovation_in_hep_workshop_hongkong.txt

}


presentation-preprocessor-args-full(){ printf "%s\n" -DFULL ; } 
presentation-preprocessor-args-smry(){ printf "%s\n" -DSMRY ; } 
presentation-preprocessor-args(){  [ -n "$SMRY" ] && echo $(presentation-preprocessor-args-smry) || echo $(presentation-preprocessor-args-full) ; } 

presentation-oname-smry(){
    local iname=$(presentation-iname)
    case $iname in 
       opticks_gpu_optical_photon_simulation_nov2019_chep)      echo opticks_oct2019_dance ;; 
       opticks_gpu_optical_photon_simulation_nov2019_chep_TALK) echo opticks_oct2019_dance_TALK ;; 
                                                             *) echo $iname ;;
    esac
}
presentation-oname-full(){
    local iname=$(presentation-iname)
    case $iname in 
       opticks_gpu_optical_photon_simulation_nov2019_chep)      echo opticks_nov2019_chep ;; 
       opticks_gpu_optical_photon_simulation_nov2019_chep_TALK) echo opticks_nov2019_chep_TALK ;; 
                                                             *) echo $iname ;;  
    esac
}

presentation-oname-(){  
   : SMRY envvar switches between smry and full names
   [ -n "$SMRY" ] && echo $(presentation-oname-smry) || echo $(presentation-oname-full) ; 
}
presentation-oname(){  
   : TALK envvar being defined appends _TALK to the oname 
   local oname=$(presentation-oname-)
   [ -n "$TALK" -a "${oname: -5}" != "_TALK" ]  && echo ${oname}_TALK || echo $oname 
}

presentation-info(){ cat << EOI

    presentation-iname       : $(presentation-iname)
    presentation-oname       : $(presentation-oname)

    presentation-path        : $(presentation-path)
    presentation-url-remote  : $(presentation-url-remote)
    presentation-url-local   : $(presentation-url-local)
    presentation-dir         : $(presentation-dir)
    presentation-bdir        : $(presentation-bdir)

    PAGE : $PAGE
    SMRY : $SMRY
    TALK : $TALK

    Setting SMRY input envvar switches from the default full args to smry ones
                
    Setting TALK input envvar swithes to oname stem with _TALK appended 

        
    presentation-preprocessor-args-full : $(presentation-preprocessor-args-full)
    presentation-preprocessor-args-smry : $(presentation-preprocessor-args-smry)
    presentation-preprocessor-args      : $(presentation-preprocessor-args)

EOI
}


presentation-path(){ echo $(presentation-dir)/$(presentation-iname).txt ; }
presentation-export(){
   export PRESENTATION_INAME=$(presentation-iname)
   export PRESENTATION_ONAME=$(presentation-oname)
   export PRESENTATION_PREPROCESSOR_ARGS=$(presentation-preprocessor-args)
}
presentation-e(){ 
   cd $(presentation-dir) ; 
   local iname=$(presentation-iname)  
   vi $iname.txt s5_background_image.txt $EXTRA
}

presentation-ee(){ 
   EXTRA=opticks_20231219_using_junosw_plus_opticks_release.txt presentation-e 
}



presentation-edit(){ vi $(presentation-path) ; }
#presentation-ed(){ vi $(presentation-path) ~/workflow/admin/reps/ntu-report-may-2017.rst ; }


presentation-imake(){ 
   presentation-cd
   presentation-export
   env | grep PRESENTATION
   touch Makefile
   make PYTHON="ipython -i --"  
}

presentation-make(){
   local msg="=== $FUNCNAME :"
   presentation-cd
   echo $msg running make in PWD $PWD 
   presentation-export
   env | grep PRESENTATION

   case $(uname) in
     Linux)   make $* PYTHON=$(which python) ;;
     Darwin)  make $* ;;
   esac
     
}

presentation-make-(){
   local msg="=== $FUNCNAME :"
   echo $msg running make in PWD $PWD 
   PRESENTATION_INAME=$(presentation-iname) PRESENTATION_ONAME=$(presentation-oname) PRESENTATION_PREPROCESSOR_ARGS=$(presentation-preprocessor-args) make $*
}


presentation-writeup(){
   presentation-cd
   vi opticks_writeup.rst
}


presentation-url-local(){ echo http://localhost/env/presentation/$(presentation-oname).html?page=${1:-0} ; }
presentation-furl-local(){ echo file:///usr/local/simoncblyth.github.io/env/presentation/$(presentation-oname).html?page=${1:-0} ; }

presentation-open(){ presentation-open-$(uname) $* ; }
presentation-open-Linux(){
   open $(presentation-furl-local)
}

presentation-open-Darwin(){



   open -a Safari $(presentation-url-local $*)
   sleep 0.3
   slides-
   slides-safari    ## just resizes browser
} 

presentation-openc(){
   open -a "Google Chrome" $(presentation-url-local $*)
   sleep 0.3
   slides-
   slides-chrome   ## just resizes browser
}

presentation-remote(){       echo simoncblyth.bitbucket.io ; }
presentation-url-remote(){   echo http://$(presentation-remote)/env/presentation/$(presentation-oname).html?page=${1:-0} ; }
presentation-remote-open(){  open $(presentation-url-remote $*) ; }

presentation--(){
   : ~/env/presentation/presentation.bash
   presentation-
   presentation-info
   presentation-make
   [ $? -ne 0 ] && echo $BASH_SOURCE $FUNCNAME ERROR FROM presentation-make CHECK PYTHON ENVIRONMENT - NEED docutils && return

   local p=${P:-0}
   presentation-open ${PAGE:-$p}
   presentation-rst2talk
}

p--(){
  : ~/env/presentation/presentation.bash
  local p=${1:-0}
  P=${P:-$p} presentation--  
  : argument is default page, but envvar P has precedence
}
p-vi(){
  : ~/env/presentation/presentation.bash
  presentation-vi  
}
p-e(){
  : ~/env/presentation/presentation.bash
  presentation-e  
}
p-cd(){
  : ~/env/presentation/presentation.bash
  presentation-cd 
}
p-index(){
  : ~/env/presentation/presentation.bash
  presentation-index 
}

p-pull(){ sed 's/:/#/' << EOC
: ~/env/presentation/presentation.bash
:
: The below files are generated in two situations:
:
: 0) on workstation A when adding snapshot references and other text content to slides
: 1) on laptop Z when doing later slide editing 
:
: Due to this double generation there is strong potential for tedious merge problems 
: resulting from "git pull". To avoid that this bash function emits commands 
: that discard local changes prior to pulling.
: 
: REMEMBER TO PIPE TO SHELL TO DO THEM
:
: NB these are not sources from the point of view of env repo
: they are derived or simply copied from ~/env/presentation
: into the /usr/local/simoncblyth.github.io repo.

cd /usr/local/simoncblyth.github.io
git checkout env/presentation/$(presentation-iname).html
git checkout env/presentation/$(presentation-iname).txt
git checkout env/presentation/s5_background_image.txt
git checkout env/presentation/my_s5defs.txt

git pull 

EOC

}






presentation--2(){
   local msg="=== $FUNCNAME :"
   presentation-
   presentation-info
   presentation-cd

   case $(presentation-iname) in 
      *_TALK) echo $msg ERROR iname ends with _TALK && return 1 ;; 
   esac  

   : make standard html presentation 
   INAME=$(presentation-iname)      presentation-make-

   : open standard html presentation 
   INAME=$(presentation-iname)      presentation-open ${page:-0}

   : create _TALK.txt RST document with extra TALK pages 
   INAME=$(presentation-iname)      presentation-rst2talk

   : make annotated html presentation 
   INAME=$(presentation-iname)_TALK presentation-make-

   : open annotated html presentation 
   INAME=$(presentation-iname)_TALK presentation-open ${page:-0}

}



presentation--t(){  TALK=1 presentation-- ; }
presentation--s(){  SMRY=1 presentation-- ; }
presentation--st(){ TALK=1 SMRY=1 presentation-- ; }

presentation-rst2talk(){ 

   local msg="=== $FUNCNAME :"
   local path=$(presentation-iname).txt
   presentation-c

   case $(presentation-iname) in
      *_TALK) echo $msg nothing to do ;; 
           *) presentation-rst2talk- $path ;;
   esac    

}

presentation-rst2talk-(){
   local msg="=== $FUNCNAME :"
   local path=$1
   local cmd="${PYTHON:-python} $(which rst2rst.py) $path" 
   echo $msg cmd $cmd
   eval $cmd
}





#presentation-iname(){ echo gpu_accelerated_geant4_simulation ; }
#presentation-iname(){ echo optical_photon_simulation_with_nvidia_optix ; }
#presentation-iname(){ echo optical_photon_simulation_progress ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_psroc ; }

#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_march2016 ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_april2016_gtc ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_may2016_lecospa ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jul2016_weihai ; }
#presentation-iname(){ echo jnu_cmake_ctest ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_oct2016_chep ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_nov2016_llr ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jan2017_psroc ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jul2017_ihep ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_sep2017_jinan ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_sep2017_wollongong ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jul2018_chep ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jul2018_ihep ; }    ## LOTS OF ISSUES : HAS OWN s5_background_image list 
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_sep2018_qingdao ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_oct2018_ihep ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jan2019_sjtu ; }   ## REMOVED INDIV s5_background_image list 
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_jul2019_ihep ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_oct2019_dance ; }  ## NOT FOUND

#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_nov2019_chep ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_nov2019_chep_TALK ; }

#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_dec2019_ihep_epd_seminar ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_dec2019_ihep_epd_seminar_TALK ; }

#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_dec2019_gtc_china_suzhou ; }
#presentation-iname(){ echo opticks_gpu_optical_photon_simulation_dec2019_gtc_china_suzhou_TALK ; }

#presentation-iname(){ echo ${INAME:-opticks_may2020_hsf} ; }

#presentation-iname(){ echo ${INAME:-opticks_jul2020_juno} ; }   ## PROBLEMS

#presentation-iname(){ echo ${INAME:-opticks_aug2020_sjtu_neutrino_telescope_workshop} ; }
#presentation-iname(){ echo ${INAME:-opticks_aug2020_sjtu_neutrino_telescope_workshop_TALK} ; }

# instead of manually changing to _TALK use the p2.sh script

#presentation-iname(){ echo ${INAME:-opticks_jan2021_juno_sim_review} ; }

#presentation-iname(){  echo ${INAME:-lz_opticks_optix7} ; }  ## NOT FOUND
#presentation-iname(){  echo ${INAME:-lz_opticks_optix7_20210208} ; }
#presentation-iname(){  echo ${INAME:-lz_opticks_optix7_20210225} ; }

#presentation-iname(){  echo ${INAME:-opticks_detector_geometry_caf_mar2021} ; }
#presentation-iname(){  echo ${INAME:-opticks_detector_geometry_caf_mar2021_TALK} ; }

#presentation-iname(){  echo ${INAME:-lz_opticks_optix7_20210315} ; }
#presentation-iname(){  echo ${INAME:-lz_opticks_optix7_20210406} ; }

#presentation-iname(){  echo ${INAME:-juno_opticks_20210426} ; }
#presentation-iname(){   echo ${INAME:-lz_opticks_optix7_20210504} ; }
#presentation-iname(){   echo ${INAME:-opticks_vchep_2021_may19} ; }
#presentation-iname(){   echo ${INAME:-lz_opticks_optix7_20210518} ; }


#presentation-iname(){  echo ${INAME:-juno_opticks_20210712} ; }
#presentation-iname(){   echo ${INAME:-lz_opticks_optix7_20210727} ; }

#presentation-iname(){  echo ${INAME:-juno_opticks_cerenkov_20210902} ; }
#presentation-iname(){  echo ${INAME:-opticks_autumn_20211019} ; }
#presentation-iname(){  echo ${INAME:-opticks_20211117} ; }


#presentation-iname(){  echo ${INAME:-opticks_20211223_pre_xmas} ; }
#presentation-iname(){ echo ${INAME:-opticks_20220115_innovation_in_hep_workshop_hongkong} ; }
#presentation-iname(){ echo ${INAME:-opticks_20220115_innovation_in_hep_workshop_hongkong_TALK} ; }
#presentation-iname(){ echo ${INAME:-opticks_20220118_juno_collaboration_meeting} ; }
#presentation-iname(){ echo ${INAME:-opticks_20220118_juno_collaboration_meeting_TALK} ; }
#presentation-iname(){  echo ${INAME:-opticks_20220227_LHCbRich_UK_GPU_HACKATHON} ; }
#presentation-iname(){  echo ${INAME:-opticks_20220307_fixed_global_leaf_placement_issue} ; }

#presentation-iname(){  echo ${INAME:-opticks_20220329_progress_towards_production} ; }
#presentation-iname(){  echo ${INAME:-opticks_20220718_towards_production_use_juno_collab_meeting} ; }
#presentation-iname(){  echo ${INAME:-opticks_202209XX_mask_spurious_debug} ; }
#presentation-iname(){  echo ${INAME:-opticks_20221117_mask_debug_and_tmm} ; }
#presentation-iname(){  echo ${INAME:-opticks_20221220_junoPMTOpticalModel_FastSim_issues_and_CustomG4OpBoundaryProcess_fix} ; }
#presentation-iname(){   echo ${INAME:-opticks_20230206_JUNO_PMT_Geometry_and_Optical_Model_Progress} ; }
#presentation-iname(){   echo ${INAME:-opticks_202303XX_jPOM_issues_and_CustomG4OpBoundaryProcess_validation} ; }
#presentation-iname(){    echo ${INAME:-opticks_202303XX_More_junoPMTOpticalModel_issues_and_Validation_of_CustomG4OpBoundaryProcess_fix} ; }
#presentation-iname(){    echo ${INAME:-opticks_20230428_More_junoPMTOpticalModel_issues_and_Validation_of_CustomG4OpBoundaryProcess_fix} ; }
#presentation-iname(){   echo ${INAME:-opticks_20230508_chep} ; }
#presentation-iname(){ echo ${INAME:-opticks_20230525_MR180_timestamp_analysis} ; }
#presentation-iname(){ echo ${INAME:-opticks_20230611_qingdao_sdu_workshop} ;  }
#presentation-iname(){ echo ${INAME:-opticks_20230721_kaiping_pwg_afg} ; }
#presentation-iname(){ echo ${INAME:-opticks_20230726_kaiping_software_review} ; }
#presentation-iname(){ echo ${INAME:-opticks_202309XX_3inch_fix} ; }
#presentation-iname(){ echo ${INAME:-opticks_20230907_release} ; }
#presentation-iname(){ echo ${INAME:-opticks_202310XX_release} ; }
#presentation-iname(){ echo ${INAME:-standalone_20230930_cpp_test_debug_ana_with_numpy} ; }
#presentation-iname(){ echo ${INAME:-opticks_20231027_nanjing_cepc_workshop} ; }
#presentation-iname(){ echo ${INAME:-opticks_20231211_profile} ; }
#presentation-iname(){ echo ${INAME:-opticks_20231219_using_junosw_plus_opticks_release} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240215_geant4_forum} ; }

#presentation-iname(){ echo ${INAME:-opticks_202401XX_next} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240224_offline_software_review} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240227_zhejiang_seminar} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240227_zhejiang_seminar_TALK} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240418_ihep_epd_seminar_story_of_opticks} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240606_ihep_panel_30min} ; }


#presentation-iname(){ echo ${INAME:-opticks_202406XX_kaiping_status_and_plan} ; }
#presentation-iname(){ echo ${INAME:-opticks_20240702_kaiping_status_and_plan} ; }
#presentation-iname(){ echo ${INAME:-opticks_20241021_krakow_chep2024} ; }
#presentation-iname(){ echo ${INAME:-opticks_20241025_montreal_nEXO_light_simulations_workshop} ; }
#presentation-iname(){ echo ${INAME:-opticks_20241025_montreal_nEXO_light_simulations_workshop} ; }
#presentation-iname(){ echo ${INAME:-opticks_20250115_kaiping_v2} ; }
#presentation-iname(){ echo ${INAME:-cpu_gpu_production} ; }
#presentation-iname(){ echo ${INAME:-opticks_20241122_ihep_assessment_5min} ; }
#presentation-iname(){ echo ${INAME:-opticks_20241122_ihep_assessment_5min_TALK} ; }
#presentation-iname(){ echo ${INAME:-opticks_20250303_ihep_gpu_symposium} ; }
#presentation-iname(){ echo ${INAME:-opticks_20250708} ; }
#presentation-iname(){ echo ${INAME:-opticks_20250727_kaiping} ; }
#presentation-iname(){ echo ${INAME:-opticks_202509YY} ; }
#presentation-iname(){ echo ${INAME:-opticks_20250917_dirac_workshop_ihep} ; }

#presentation-iname(){ echo ${INAME:-opticks_20251208_ihep_assessment_5min} ; }
presentation-iname(){ echo ${INAME:-opticks_20251219_geometry_simtrace_check} ; }


