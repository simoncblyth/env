# === func-gen- : windows/vs/vs fgp windows/vs/vs.bash fgn vs fgh windows/vs
vs-src(){      echo windows/vs/vs.bash ; }
vs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vs-src)} ; }
vs-vi(){       vi $(vs-source) ; }
vs-env(){      elocal- ; }
vs-usage(){ cat << \EOU

Microsoft Visual Studio 2015 Community Edition
================================================

Windows Development Environment
---------------------------------

* :google:`Windows Development Environment`

* https://github.com/felixrieseberg/windows-development-environment

Install Chocolatey
~~~~~~~~~~~~~~~~~~~

* see chocolatey-

Find Command Prompt (Windows 7 SP1)
-------------------------------------

* Start Menu, triangle expand "All Programs"
* right click "Visual Studio 2015" and Expand, and again on "Visual Studio Tools"

::

    Developer Command Prompt for VS2015
    MSBuild Command Prompt 


Standard Windows cmd.exe
--------------------------

* windows-key R (for Run menu)
* type "cmd" and ok

From cmd.exe shell entering @powershell gets you there.


Run cmd.exe as administrator
------------------------------

* Start > All Programs > Accessories
* Right click "Command Prompt" and choose "Run as administrator" answer Y to user access control question, 
* this gets to a prompt that looks exactly like non-admin one




cl compiler
-------------

* all three of above prompts didnt recognize cl 

git 
----

* seems git can be used from IDE, but not exposed from shells

powershell git : posh-git
---------------------------

* https://github.com/dahlbyk/posh-git




EOU
}
vs-dir(){ echo $(local-base)/env/windows/vs/windows/vs-vs ; }
vs-cd(){  cd $(vs-dir); }
vs-mate(){ mate $(vs-dir) ; }
vs-get(){
   local dir=$(dirname $(vs-dir)) &&  mkdir -p $dir && cd $dir

}
