# === func-gen- : scm/bitbucket/bitbucket fgp scm/bitbucket/bitbucket.bash fgn bitbucket fgh scm/bitbucket
bitbucket-src(){      echo scm/bitbucket/bitbucket.bash ; }
bitbucket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bitbucket-src)} ; }
bitbucket-vi(){       vi $(bitbucket-source) ; }
bitbucket-env(){      elocal- ; }
bitbucket-usage(){ cat << EOU

BITBUCKET
==========

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

Add bbenv target to ~/env/Makefile that populates ~/simoncblyth.bitbucket.org/env with 
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

Most of the binaries are not huge, only the video is potentially a problem, 
maybe dropbox for that.


video on dropbox
------------------

* https://tech.dropbox.com/2014/02/video-processing-at-dropbox/
* http://eastasiastudent.net/china/dropbox-no-vpn
* http://techcrunch.com/2014/02/17/dropbox-now-accessible-for-the-first-time-in-china-since-2010/

Some suggestions to add to /etc/hosts::

    174.36.30.73 www.dropbox.com
    174.36.30.71 www.dropbox.com


resource collection
---------------------

#. Extended ~/e/muon_simulation/presentation/rst2s5-2.6.py to doctree traverse collecting 
   and resolving the urls of resources used in the document (images, videos, background images).



EOU
}
bitbucket-dir(){ echo $(local-base)/env/scm/bitbucket/scm/bitbucket-bitbucket ; }
bitbucket-cd(){  cd $(bitbucket-dir); }
bitbucket-mate(){ mate $(bitbucket-dir) ; }
bitbucket-get(){
   local dir=$(dirname $(bitbucket-dir)) &&  mkdir -p $dir && cd $dir

}
