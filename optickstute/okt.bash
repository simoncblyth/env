# === func-gen- : optickstute/okt fgp optickstute/okt.bash fgn okt fgh optickstute
okt-src(){      echo optickstute/okt.bash ; }
okt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(okt-src)} ; }
okt-vi(){       vi $(okt-source) ; }
okt-env(){      elocal- ; }
okt-usage(){ cat << EOU

Opticks Tutorial Preparation Notes
===================================

TODO
----

Populate the docs http://simoncblyth.bitbucket.org/opticks/

* proj descriptions from the bash usage functions
  these are currently mostly devnotes, separate those out 
  and replace with short user descriptions 



Suggestions from Tao
-----------------------

::

    Hi Simon,

    If I am a student, I would like to:

    * Have a quick look at the application self. The visualization is more attractive. So I need to know:
      * basic commands to control the software self.
      * what different events would look like.
        * how to specify an input file.
      * options to control the Opticks.

    * Is that possible to visualize detectors from different experiment.
      * If I am not familiar with Geant4, I would like to use an example from Geant4.
        * how to export GDML/DAE
        * how to import it to Opticks

    * Then is that possible to install the software in my own desktop/laptop with GPU.
      * The hardware requirements.

    * The architecture of Opticks.
    * How to develop the program.

    Simon, could you send me your report/tutorial to me if they are ready?
    Thanks!

    Tao


Inspiration
-------------

* http://www-public.slac.stanford.edu/geant4/PastEvents.asp


OpenMesh Tutorial content 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00066.html


What to get users to do ?
--------------------------

* need code samples doing different things

* need analysis of simulated photon data, using numpy 
  to make some plots



Ideas
------

* installation screen capture video 


Installation Questions
------------------------

* fraction of users with machine, breakdown Linux/macOS/Windows ? What versions ?
* how many CUDA capable GPU machines ?







EOU
}
okt-dir(){ echo $(local-base)/env/optickstute/optickstute-okt ; }
okt-cd(){  cd $(okt-dir); }
okt-mate(){ mate $(okt-dir) ; }
okt-get(){
   local dir=$(dirname $(okt-dir)) &&  mkdir -p $dir && cd $dir

}
