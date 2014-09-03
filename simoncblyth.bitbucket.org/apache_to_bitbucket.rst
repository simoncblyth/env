Apache to bitbucket 
====================

.. contents:: :local:

Overview
---------

Notes on migrating storing static resources (html pages, images, PDF, video)
from locally hosted apache to cloud servers from bitbucket and dropbox.

Related
--------

* :doc:`/simoncblyth.bitbucket.org/index`


Creation 
-----------

https://confluence.atlassian.com/display/BITBUCKET/Publishing+a+Website+on+Bitbucket

#. With bitbucket web interface create repository named: *simoncblyth.bitbucket.org*
#. Clone that repo locally::

    delta:~ blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/simoncblyth.bitbucket.org
    destination directory: simoncblyth.bitbucket.org
    no changes found
    updating to branch default
    0 files updated, 0 files merged, 0 files removed, 0 files unresolved
    delta:~ blyth$ 

#. add, commit, push static index.html to the root


RST doc links
--------------

Absolute Sphinx RST doc links need to omit the **/env/notes** 
prefix that arises due to the positioning of the root index.html.

For example::

   :doc:`/simoncblyth.bitbucket.org/index`


* :doc:`/simoncblyth.bitbucket.org/index`

env notes
-----------


Add **bb** target to ~/env/Makefile that populates ~/simoncblyth.bitbucket.org/env with 
the Sphinx generated html.

#. slightly funny doing that out of svn working copy, but have not migrated yet, still testing  

::

    delta:simoncblyth.bitbucket.org blyth$ hg commit -m "test Sphinx html with bitbucket static pages "
    delta:simoncblyth.bitbucket.org blyth$ 
    delta:simoncblyth.bitbucket.org blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/simoncblyth.bitbucket.org
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes


Bitbucket Static pages rejig
-----------------------------

C2 is again offline, last straw


#. through bitbucket web interface 

   * delete repository named *simoncblyth.bitbucket.org*  
   * create repository named *simoncblyth.bitbucket.org*   # make it public this time

#. in file system::

      delta:~ blyth$ rm -rf simoncblyth.bitbucket.org    # delete old clone
      delta:~ blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/simoncblyth.bitbucket.org   # clone the empty

#. amend **bb** target of env/Makefile Sphinx build to create/populate BITBUCKET_HTDOCS/env/notes and build::

      delta:e blyth$ make bb 

#. amend target of env/muon_simulation/presentation/Makefile to create populate BITBUCKET_HTDOCS/env/muon_simulation/presentation and build::

      slides-;slides-scd;make  # OR slides-;slides-make

#. move original APACHE_HTDOCS/env to env.keep and create symbolic link to allow local apache 
   testing of html and resources without committing+pushing to bitbucket::

    delta:Documents blyth$ sudo ln -s /Users/blyth/simoncblyth.bitbucket.org/env  

#. move across the contents of APACHE_HTDOCS/env.keep to the new BITBUCKET_HTDOCS/env 
   excluding the videos

#. add to the bitbucket repo, mercurial complains about big PDF, so revert and place in Dropbox/Public

    env/muon_simulation/presentation/gpu_optical_photon_simulation.pdf: up to 115 MB of RAM may be required to manage this file
    (use 'hg revert env/muon_simulation/presentation/gpu_optical_photon_simulation.pdf' to cancel the pending addition)

    delta:simoncblyth.bitbucket.org blyth$ du -h env/muon_simulation/presentation/gpu_optical_photon_simulation.pdf
     37M    env/muon_simulation/presentation/gpu_optical_photon_simulation.pdf

    (adm_env)delta:simoncblyth.bitbucket.org blyth$ mv env/muon_simulation/presentation/gpu_optical_photon_simulation.pdf ~/Dropbox/Public/



Layout migration rejig
-----------------------

Sphinx derived html, for env at least, are very much **notes**. 
Make that explicit and avoid double top "e" and "env" with 
layout at expense of the path.  As all repos share the one 

#. /env/notes   Sphinx derived notes
#. /env/...     other resources
#. /env/muon_simulation/presentation/

Formerly Sphinx and slides building machinery generates html 
into APACHE_HTDOCS/e and APACHE_HTDOCS/env respectively. 
Instead of this generate into BITBUCKET_HTDOCS/env/notes and BITBUCKET_HTDOCS/env
Then can publish by a Mercurial commit and push.

All bitbucket repos under a username share the same single static pages repo, 
so having a one to one correspondence to a top level dir named after 
the repo is the cleanest way. 

Whats missing
---------------

On cms02, also used bare apache for docs like presentations and images::

    find $(apache-htdocs)/env

Need way to reference those from Sphinx pages, and need to avoid the big ones::

    delta:env blyth$ du -hs $(apache-htdocs)/env
    2.4G    /Library/WebServer/Documents/env

Bitbucket limits


* https://confluence.atlassian.com/pages/viewpage.action?pageId=273877699
* http://stackoverflow.com/questions/1284669/how-do-i-manage-large-art-assets-appropriately-in-dvcs

TODO: Investigate Dropbox for longterm holding of binaries


Sphinx downloads
-----------------

For small numbers of binaries can use Sphinx download with RST source like::

    described in the :download:`Chroma whitepaper <chroma.pdf>`.

Not so keen on this, 

#. it results in having multiple copies of the binary that 
   get copied around by the Sphinx build.  

#. Prefer a single resource with a single URL, that never gets copied


existing resource approach and how to map into bitbucket
----------------------------------------------------------

#. binaries (images, pdf, videos) served by apache from $APACHE_HTDOCS/env/
#. Sphinx derived html served from $APACHE_HTDOCS/e/ 

Keeping big and unchanging binaries separate from 
small and frequently changing html is a good pattern to continue.  
Could map this into bitbucket via directory structure in the static
pages repo::

   /var/scm/mercurial/simoncblyth.bitbucket.org/e/ 
   /var/scm/mercurial/simoncblyth.bitbucket.org/env/ 

But its all one repo anyhow, that is populated by 

#. sphinx build
#. manual placement of resources 

Could in principal create a script to merge 
the resource and derived html trees ? But that introduces 
complication and makes it difficult to do clean Sphinx builds.

managing the binaries
-----------------------

Most of the binaries are not huge, only the video is potentially a problem, 
maybe dropbox for that.  But need a way to select only binaries that are 
actually referred to.

dropbox alternatives
---------------------

* https://www.yunio.com

video on dropbox
------------------

* https://tech.dropbox.com/2014/02/video-processing-at-dropbox/
* http://eastasiastudent.net/china/dropbox-no-vpn
* http://techcrunch.com/2014/02/17/dropbox-now-accessible-for-the-first-time-in-china-since-2010/

Right click on video stored in your Public Dropbox folder to get the Public link, include
that URL in 

Some suggestions to add to /etc/hosts::

    174.36.30.73 www.dropbox.com
    174.36.30.71 www.dropbox.com

resource collection
---------------------

#. Extended ~/e/muon_simulation/presentation/rst2s5-2.6.py to doctree traverse collecting 
   and resolving the urls of resources used in the document (images, videos, background images).

#. ~/e/bin/resources.py adding up sizes


https://www.dropbox.com/s/6jmcqxphnc8qhkg/g4daeview_001.m4v?dl=0



