# === func-gen- : base/macports fgp base/macports.bash fgn macports fgh base
macports-src(){      echo base/macports.bash ; }
macports-source(){   echo ${BASH_SOURCE:-$(env-home)/$(macports-src)} ; }
macports-vi(){       vi $(macports-source) ; }
macports-usage(){ cat << EOU

MACPORTS
===========

::

       port installed
       port list installed     ## not the same as above, and much slower 


Package Manager Installation from source distribution
--------------------------------------------------------

Follow along http://www.macports.org/install.php


Xcode Developer Tools Version Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires Xcode Developer Tools (version 4.4 or later for Mountain Lion, 
4.1 or later for Lion, 3.2 or later for Snow Leopard, or 3.1 or later for Leopard)

b2mc uses Lion with Xcode 4.3.1::

	b2mc:MacPorts-2.1.3 heprez$ uname -a
	Darwin b2mc.kek.jp 11.4.0 Darwin Kernel Version 11.4.0: Mon Apr  9 19:32:15 PDT 2012; root:xnu-1699.26.8~1/RELEASE_X86_64 x86_64

	b2mc:MacPorts-2.1.3 heprez$ mdls -name kMDItemVersion /Applications/XCode.app
	kMDItemVersion = "4.3.1"


Xcode EULA agreement
~~~~~~~~~~~~~~~~~~~~~~

/Applications/Xcode.app/Contents/Resources/English.lproj/License.rtf

::

	b2mc:MacPorts-2.1.3 heprez$ xcodebuild -license

	You have not agreed to the Xcode license agreements. You must agree to both license agreements below in order to use Xcode.
	Hit the Enter key to view the license agreements at '/Applications/Xcode.app/Contents/Resources/English.lproj/License.rtf'
	...

	By typing 'agree' you are agreeing to the terms of the software license agreements. Type 'print' to print them or anything else to cancel, [agree, print, cancel]  agree

	You can view the license agreements in Xcode's About Box, or at /Applications/Xcode.app/Contents/Resources/English.lproj/License.rtf

Source Build
~~~~~~~~~~~~~~~

::

        ./configure && make && sudo make install

::

	Congratulations, you have successfully installed the MacPorts system. To get the Portfiles and update the system, add /opt/local/bin to your PATH and run:
	sudo port -v selfupdate
	Please read "man port", the MacPorts guide at http://guide.macports.org/ and Wiki at https://trac.macports.org/ for full documentation. 


selfupdate
~~~~~~~~~~~~

After adding /opt/local/bin to PATH in ~/.bash_profile::

	b2mc:MacPorts-2.1.3 heprez$ sudo port -v selfupdate
	--->  Updating MacPorts base sources using rsync
	receiving file list ... done
	base.tar


::

    simon:e blyth$ date
    Mon  9 Sep 2013 15:26:11 CST
    simon:e blyth$ sudo port selfupdate
    Password:
    --->  Updating MacPorts base sources using rsync
    MacPorts base version 2.1.2 installed,
    MacPorts base version 2.2.0 downloaded.
    --->  Updating the ports tree
    --->  MacPorts base is outdated, installing new version 2.2.0
    Installing new MacPorts release in /opt/local as root:admin; permissions 0755; Tcl-Package in /Library/Tcl
    The ports tree has been updated. To upgrade your installed ports, you should run
      port upgrade outdated




Selecting ports
-----------------

See *port help select*, usage example::

	  port select --list python 
	  Available versions for python:
		none
		python25
		python25-apple
		python26 (active)

	  sudo port select python python25         ##  python 2.5.5
	  sudo port select python python25-apple   ##  python 2.5.1  still /opt/local/bin/python
	  sudo port select python none             ##  python 2.5.1  direct /usr/bin/python

	  port select --show python 

Functions
-----------

*macports-get*
          source get, typically GUI approaches are used rather than this commandline way 

          http://www.macports.org/install.php


EOU
}
macports-dir(){ echo $(local-base)/env/base/$(macports-name) ; }
macports-cd(){  cd $(macports-dir); }
macports-name(){ echo MacPorts-2.1.3 ; }
macports-mate(){ mate $(macports-dir) ; }
macports-get(){
   local dir=$(dirname $(macports-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(macports-name)
   local tgz=$nam.tar.gz
   local url=https://distfiles.macports.org/MacPorts/$tgz

   [ ! -f "$tgz" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && tar zxvf $tgz 

}

macports-clean(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo port clean --all all"
   echo $msg $cmd
   eval $cmd
}

macports-space(){
   local space=space.txt
   [ ! -f $space ] && sudo port space installed > $space
   grep MiB $space | sort -g -r | head -40
}


macports-env(){
   elocal- 
   ## avoid stomping on the virtualenv
   if [ -z "$VIRTUAL_ENV" ]; then
       export PATH=/opt/local/bin:/opt/local/sbin:$PATH
       export MANPATH=/opt/local/share/man:$MANPATH
   fi
}


