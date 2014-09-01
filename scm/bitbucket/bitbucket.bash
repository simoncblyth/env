# === func-gen- : scm/bitbucket/bitbucket fgp scm/bitbucket/bitbucket.bash fgn bitbucket fgh scm/bitbucket
bitbucket-src(){      echo scm/bitbucket/bitbucket.bash ; }
bitbucket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bitbucket-src)} ; }
bitbucket-vi(){       vi $(bitbucket-source) ; }
bitbucket-env(){      elocal- ; }
bitbucket-usage(){ cat << EOU

BITBUCKET
==========

related
--------

* see *hg-* regarding the conversion from Subversion to Mercurial


Upload env Mercurial conversion to bitbucket
----------------------------------------------

#. create repo with bitbucket webinterface named "env", leave it private for now
#. adjust paths in the bare converted repo on D

::

    delta:~ blyth$ cd /var/scm/mercurial/env/.hg
    delta:.hg blyth$ vi hgrc     # this is a bare converted repo, so this is creating the file
    delta:.hg blyth$ cd ..
    delta:env blyth$ hg paths
    default = ssh://hg@bitbucket.org/simoncblyth/env

#. push up to bitbucket, took under 2 minutes

::

    delta:env blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/env
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 4621 changesets with 13955 changes to 4349 files


#. observations https://bitbucket.org/simoncblyth/env

   * cursory glance suggests all commits/history are there
     TODO: clone from bitbucket for systematic check 
 
   * blyth/lint/maqm : Author not mapped to a bitbucket user   
   * added and verified my gmail to use for this 


Trial Usage
------------

Use ssh not http, for auto authentication via key::

    delta:~ blyth$ mv env env.svn

    delta:~ blyth$ hg clone https://simoncblyth@bitbucket.org/simoncblyth/env
    http authorization required
    realm: Bitbucket.org HTTP
    user: simoncblyth
    password: interrupted!

    delta:~ blyth$ hg clone ssh://hg@bitbucket.org/simoncblyth/env
    destination directory: env
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 4636 changesets with 14033 changes to 4363 files
    updating to branch default
    3051 files updated, 0 files merged, 0 files removed, 0 files unresolved



Username Mapping
-------------------

https://confluence.atlassian.com/display/BITBUCKET/Set+your+username+for+Bitbucket+actions

Bitbucket requires that the email address you commit with matches a validated
email address on an account. On your local system, where you commit, both Git
and Mercurial allow you to configure a global username/email and a repository
specific username/email. If you do not specify a repository specific
username/email values, both systems use the global default. So, for example, if
your Bitbucket account has a validated email address of joe.foot@gmail.com, you
need to make sure your repository configuration is set for that username. Also,
make sure you have set your global username/email address configuration to a
validated email address.

Checking raw commits with bitbucket web interface shows no associated email address::

   # HG changeset patch
   # User blyth

*hg convert* has a usermap option, so can setup mapping from SVN users like
"blyth" to an email address verified with bitbucket. 


Bitbucket Teams
----------------

* https://confluence.atlassian.com/display/BITBUCKET/Teams+Frequently+Asked+Questions


Bitbucket Static Pages
------------------------

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
    remote: added 1 changesets with 942 changes to 942 files


#. Pages appear at http://simoncblyth.bitbucket.org/env/


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


Bitbucket Static pages index 
------------------------------

* http://simoncblyth.bitbucket.org 

Machinery to generate index html from RST sources 
with links to the notes and presentations::

    cd ~/env/scm/bitbucket/simoncblyth.bitbucket.org    
          # OR bitbucket-;bitbucket-scd
    make       
          # writes ~/simoncblyth.bitbucket.org/index.html

    cd ~/simoncblyth.bitbucket.org                      
          # bitbucket-cd

    hg commit -m "update top index "
    hg push
    open http://simoncblyth.bitbucket.org


Notes TODO
-----------

* http://simoncblyth.bitbucket.org/env/notes/ 

* currently using dirhtml layout, 

  * problematic for offline html use without a webserver
  * BUT nice short URLs 
  * maybe leap to plain html style like workflow does

Slides TODO
------------

* http://simoncblyth.bitbucket.org/env/muon_simulation/presentation/gpu_optical_photon_simulation.html

#. trac links need updates to bitbucket ones, after make leap to env in bitbucket and open env repo 
#. video and presentation PDF are too big for sensible storage in bitbucket repo, need
   to find another home (Dropbox ?) and link to it 

   * linking video from Dropbox, works but very slow

Dropbox
-------

#. problem is that the dropbox is designed to be everywhere, 
   including devices with very little available space having big files 
   on those makes no sense

   * create a new dropbox account and only use its web interface for uploading  


::

    delta:Dropbox blyth$ mv /Library/WebServer/Documents/env.keep/g4daeview_001.m4v Public/

   * https://www.dropbox.com/s/6jmcqxphnc8qhkg/g4daeview_001.m4v?dl=1  video preview html page and over compressed video
   * https://www.dropbox.com/s/6jmcqxphnc8qhkg/g4daeview_001.m4v?dl=0  good quality video, but slow to load 
   * https://dl.dropboxusercontent.com/u/53346160/g4daeview_001.m4v     # hmm different ways of getting the link yield differet links 


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


bitbucket paths
----------------

After cloning /var/scm/mercurial/env into /tmp/t/env the
/tmp/t/env/.hg/hgrc has a paths setting that points back to where it was 
cloned from

::

    delta:env blyth$ hg paths
    default = /var/scm/mercurial/env






EOU
}

bitbucket-repo(){  echo simoncblyth.bitbucket.org ; }
bitbucket-sdir(){ echo $(env-home)/scm/bitbucket/$(bitbucket-repo) ; }
bitbucket-scd(){  cd $(bitbucket-sdir); }
bitbucket-dir(){ echo $(bitbucket-htdocs) ; }
bitbucket-cd(){  cd $(bitbucket-dir); }
bitbucket-htdocs(){ echo $HOME/$(bitbucket-repo) ; }
bitbucket-export(){
   export BITBUCKET_HTDOCS=$(bitbucket-htdocs)
}

bitbucket-username(){ echo ${BITBUCKET_USERNAME:-simoncblyth} ; }
bitbucket-repo(){ echo /var/scm/mercurial/${1:-env} ; }

bitbucket-paths-(){ 
   local name=${1:-env}
   cat << EOC

[paths]
default = ssh://hg@bitbucket.org/$(bitbucket-username)/$name

EOC
}
bitbucket-paths(){
   local name=${1:-env}
   local repo=$(bitbucket-repo $name)
   local hgrc=$repo/.hg/hgrc

   [ ! -d "$(dirname $hgrc)" ] && echo $msg ERROR no .hg dir $(dirname $hgrc) && return
   [ ! -f "$hgrc" ] && echo $msg writing $hrgc && $FUNCNAME- $ name > $hgrc 
   [ -f "$hgrc" ] && echo $msg hgrc $hgcr && cat $hgrc 

}



