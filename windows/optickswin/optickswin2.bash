optickswin2-source(){   echo ${BASH_SOURCE} ; }
optickswin2-vi(){       vi $(optickswin2-source) ; }
optickswin2-env(){      elocal- ; }
optickswin2-usage(){ cat << \EOU

Opticks Port to windows
=========================

See also 
---------

* optickswin-
* vs-
* gitforwindows-



Git for windows (v2.18.0) installation : provides bash shell with git 
------------------------------------------------------------------------

1. download gitforwindows (may need to use alternate browser, as explorer refused to download the .exe)
2. run the .exe installer using options shown below

Location
   instead of the default C:\\Program Files\\Git 
   used powershell to create Local in order to install at C:\\Local\\Git

Style
   Picked the default:

   * (safest) Use Git from Git Bash only
   * (default) Use Git from the Windows Command Prompt 
   * Use Git and optional Unix tools from the Windows Command Prompt       

Transport
   Picked default OpenSSL (not native Windows secure channel)

Line endings
   Picked default recommended for windows: Checkout Windows-style, commit Unix-style     

Terminal
   Picked default MinTTY (default terminal of MSYS2). Windows console programs
   (such as interactive Python) must be launched via winpty to work in MinTTY

   Didnt pick: Windows default console window  

Release notes at file:///C:/Local/Git/ReleaseNotes.html


Windows developer settings
---------------------------

https://github.com/felixrieseberg/windows-development-environment

1. Enable Developer Mode (Settings - Update & security > For developers)

2. In settings, Search for “Windows Features” and choose “Turn Windows features on or off” and enable Windows Subsystem for Linux.

   * it needs a restart following this  


Now need a package manager
----------------------------

chocolatey-
nuget-


wsl : windows subsystem for linux : doesnt work with GPU yet : so drop this approach
--------------------------------------------------------------------------------------

* https://docs.microsoft.com/en-us/windows/wsl/about
* https://github.com/Microsoft/WSL/issues/829

find windows build
---------------------

Settings > System > About


Install CUDA 9.2 using NVIDIA Installer
-----------------------------------------

Options, picked the recommended one:

* Express (recommended) Installs all CUDA components and overwrites current Display Driver
* Custom : Allows to select the components to install

Installer warns that should install Visual Studio First. 


Visual Studio Community 2017
--------------------------------














EOU
}
optickswin2-dir(){ echo $(local-base)/env/windows/optickswin2/windows/optickswin2-optickswin2 ; }
optickswin2-cd(){  cd $(optickswin2-dir); }
optickswin2-mate(){ mate $(optickswin2-dir) ; }
optickswin2-get(){
   local dir=$(dirname $(optickswin2-dir)) &&  mkdir -p $dir && cd $dir

}
