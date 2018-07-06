# === func-gen- : simoncblyth.bitbucket.org/bitbucketstatic fgp simoncblyth.bitbucket.org/bitbucketstatic.bash fgn bitbucketstatic fgh simoncblyth.bitbucket.org



bitbucketstatic-src(){      echo simoncblyth.bitbucket.org/bitbucketstatic.bash ; }
bitbucketstatic-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bitbucketstatic-src)} ; }
bitbucketstatic-vi(){       vi $(bitbucketstatic-source) ; }
bitbucketstatic-env(){      elocal- ; }
bitbucketstatic-usage(){ cat << EOU

Bitbucket Static Pages 
=======================

* http://simoncblyth.bitbucket.org 

Sources/Binaries
------------------

The repo ~/simoncblyth.bitbucket.org contains:

* binaries arranged in directories
* html pages derived from sources in ~/env

Guidelines for binaries
------------------------

#. not too big there is a 1GB limit on repo size
#. use parallel folder hierarchy to the env repository  

Index Page 
-----------

Machinery to generate index html from RST sources 
with links to the notes and presentations::

    cd ~/env/simoncblyth.bitbucket.org    # OR bitbucketstatic-;bitbucketstatic-scd
    make                                  # writes ~/simoncblyth.bitbucket.org/index.html

    cd ~/simoncblyth.bitbucket.org        # bitbucketstatic-cd

    hg commit -m "update top index "
    hg push
    open http://simoncblyth.bitbucket.org

Notes 
-------

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



EOU
}
bitbucketstatic-dir(){ echo $HOME/simoncblyth.bitbucket.org ; }
bitbucketstatic-sdir(){ echo $HOME/env/simoncblyth.bitbucket.org ; }
bitbucketstatic-cd(){  cd $(bitbucketstatic-dir); }
bitbucketstatic-scd(){  cd $(bitbucketstatic-sdir); }
bitbucketstatic-make(){
   bitbucketstatic-scd
   make
}





