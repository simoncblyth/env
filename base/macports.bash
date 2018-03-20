# === func-gen- : base/macports fgp base/macports.bash fgn macports fgh base
macports-src(){      echo base/macports.bash ; }
macports-source(){   echo ${BASH_SOURCE:-$(env-home)/$(macports-src)} ; }
macports-vi(){       vi $(macports-source) $(port-source) ; }
macports-env(){
   elocal- 
   port- 

   ## avoid stomping on the virtualenv
   if [ -z "$VIRTUAL_ENV" ]; then
       export PATH=/opt/local/bin:/opt/local/sbin:$PATH
       export MANPATH=/opt/local/share/man:$MANPATH
   fi
}


macports-usage(){ cat << EOU

MACPORTS
===========

Listing Versions Installed/Available
---------------------------------------
::

    port installed
        ## fast, lists versions of currently installed ports

    port list installed     
        ## slow, lists latest versions of every version of ports installed (so it repeats a lot) 

::

    port list installed > ~/macports/port_list_installed_20jan2018.log
    port installed > ~/macports/port_installed_20jan2018.log
    ## after trimming some warnings, get same number of lines


Migration At System Upgrades : requires uninstall/install of all packages
----------------------------------------------------------------------------

* http://trac.macports.org/wiki/Migration

Thoughts:

* running this would take a very long time, better to extract the
  correct dependency order of the ports to install using a hacked version
  of the script and chunk the list up to ease debugging fails

In brief::

   port -qv installed > ~/macports/port-qv-installed.log   # list installed 
   sudo port -f uninstall installed  # uninstall all installed ports
   sudo rm -rf /opt/local/var/macports/build/*    # clean partials 

::

    curl --location --remote-name \
        https://github.com/macports/macports-contrib/raw/master/restore_ports/restore_ports.tcl
    chmod +x restore_ports.tcl
    sudo ./restore_ports.tcl ~/macports/port-qv-installed.log

* https://github.com/macports/macports-contrib/raw/master/restore_ports/restore_ports.tcl

Script using /opt/local/bin/port-tclsh::

    # Install a list of ports given in the form produced by 'port installed', in
    # correct dependency order so as to preserve the selected variants.


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


Partial clean : due to missing Portfile 
-----------------------------------------

::

    simon:~ blyth$ macports-clean
    === macports-clean : sudo port clean --all all
    Password:
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Cleaning 2Pong
    --->  Cleaning 3proxy
    ...
    --->  Cleaning hs-binary
    --->  Cleaning hs-blaze-builder
    --->  Cleaning hs-blaze-html
    --->  Cleaning hs-blaze-markup
    Error: Unable to open port: Could not find Portfile in /opt/local/var/macports/sources/rsync.macports.org/release/ports/devel/hs-boolean
    simon:~ blyth$ 
    simon:~ blyth$ port info hs-boolean
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    Error: Unable to open port: Could not find Portfile in /opt/local/var/macports/sources/rsync.macports.org/release/ports/devel/hs-boolean

Try to fix the broken port with a selfupdate, in order to complete the clean.


selfupdate 20 nov 2013, macports updatdes itself to 2.2.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:~ blyth$ sudo port -v selfupdate
    Password:
    --->  Updating MacPorts base sources using rsync
    receiving file list ... done
    ...
    --->  MacPorts base is outdated, installing new version 2.2.1
    Installing new MacPorts release in /opt/local as root:admin; permissions 0755; Tcl-Package in /Library/Tcl
    ...
    Congratulations, you have successfully upgraded the MacPorts system.

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


Freeing space
---------------

* https://trac.macports.org/wiki/howto/InstallingOlderPort

::

    simon:~ blyth$ sudo port -v uninstall inactive
    Password:
    --->  Unable to uninstall subversion-perlbindings @1.6.2_0, the following ports depend on it:
    --->    p5-svn-simple @0.27_0
    Error: port uninstall failed: Please uninstall the ports that depend on subversion-perlbindings first.
    simon:~ blyth$ 


Macports space inactive
-------------------------

::

    simon:~ blyth$ macports-space-inactive
    Port gdk-pixbuf2 does not contain any file or is not active.
    Port glib2-devel does not contain any file or is not active.
    Port gtk2 does not contain any file or is not active.
    Port liblzma does not contain any file or is not active.
    Port lighttpd does not contain any file or is not active.
    Port lzmautils does not contain any file or is not active.
    Port p5-error does not contain any file or is not active.
    Port p5-error does not contain any file or is not active.
    Port p5-locale-gettext does not contain any file or is not active.
    Port p5-locale-gettext does not contain any file or is not active.
    Port p5-locale-gettext does not contain any file or is not active.
    Port py25-setuptools does not contain any file or is not active.
    Port py26-distribute does not contain any file or is not active.
    Port py26-distribute does not contain any file or is not active.
    Port py26-distribute does not contain any file or is not active.
    Port py26-distribute does not contain any file or is not active.
    Port subversion-perlbindings does not contain any file or is not active.
    Port teTeX does not contain any file or is not active.
    Port texlive-bin-extra does not contain any file or is not active.
    441.094 MiB llvm-3.0 @3.0_4
    412.499 MiB boost @1.47.0_2
    412.499 MiB boost @1.38.0_0
    412.499 MiB boost @1.35.0_2
    248.246 MiB erlang @R14B01_1
    127.195 MiB mysql5 @5.1.50_1
    127.195 MiB mysql5 @5.1.49_0
    127.195 MiB mysql5 @5.0.81_0
    127.195 MiB mysql5 @5.0.81_0
    89.866 MiB texlive-basic @23152_1
    78.185 MiB texlive-latex @23089_0
    76.566 MiB python27 @2.7.2_4
    67.766 MiB python26 @2.6.7_4
    67.766 MiB python26 @2.6.7_3
    67.766 MiB python26 @2.6.6_1
    67.766 MiB python26 @2.6.6_0
    58.869 MiB perl5.12 @5.12.4_1
    58.869 MiB perl5.12 @5.12.3_3
    58.869 MiB perl5.12 @5.12.3_2
    55.978 MiB clang-3.0 @3.0_7
    54.368 MiB python25 @2.5.5_1
    54.368 MiB python25 @2.5.4_3
    53.890 MiB ghostscript @9.06_1
    53.890 MiB ghostscript @9.05_0
    53.890 MiB ghostscript @9.04_1
    53.890 MiB ghostscript @9.01_0
    53.890 MiB ghostscript @8.62_0
    41.986 MiB perl5.8 @5.8.9_3
    41.986 MiB perl5.8 @5.8.8_3
    41.631 MiB glib2 @2.30.3_0
    41.631 MiB glib2 @2.30.2_2
    41.631 MiB glib2 @2.30.2_1
    41.631 MiB glib2 @2.24.2_0
    41.631 MiB glib2 @2.20.2_0
    37.376 MiB texlive-bin @2012_4
    37.376 MiB texlive-bin @2011_5
    28.733 MiB ImageMagick @6.8.0-2_0
    28.733 MiB ImageMagick @6.7.3-1_0
    28.733 MiB ImageMagick @6.3.9-7_0
    28.539 MiB openmotif @2.3.3_1


Secret underscore required::

    simon:~ blyth$ sudo port uninstall boost @1.35.0_2
    Password:
    Error: port uninstall failed: Registry error: boost @1.35.0_2 not registered as installed

    simon:~ blyth$ sudo port uninstall boost@1.35.0_2
    Error: port uninstall failed: Registry error: boost @1.35.0_2 not registered as installed

    simon:~ blyth$ sudo port uninstall boost_@1.35.0_2


::

    simon:~ blyth$ sudo port installed  | grep llvm
      llvm-3.0 @3.0_4
      llvm-3.0 @3.0_11 (active)
      llvm_select @0.2_0 (active)
    simon:~ blyth$ sudo port uninstall -v llvm-3.0_@3.0_4



Functions
-----------

*macports-get*
          source get, typically GUI approaches are used rather than this commandline way 

          http://www.macports.org/install.php


EOU
}
macports-dir(){ echo $(local-base)/env/base/macports ; }
macports-cd(){  cd $(macports-dir); }


macports-list-save()
{
   port -qv installed 
}


macports-get-restore-ports()
{    
    local dir=$(macports-dir) &&  mkdir -p $dir 
    macports-cd
    curl --location --remote-name \
        https://github.com/macports/macports-contrib/raw/master/restore_ports/restore_ports.tcl
}


#macports-name(){ echo MacPorts-2.1.3 ; }
# 
#macports-get(){
#   local dir=$(dirname $(macports-dir)) &&  mkdir -p $dir && cd $dir
#
#   local nam=$(macports-name)
#   local tgz=$nam.tar.gz
#   local url=https://distfiles.macports.org/MacPorts/$tgz
#
#   [ ! -f "$tgz" ] && curl -L -O "$url"
#   [ ! -d "$nam" ] && tar zxvf $tgz 
#
#}

macports-clean(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo port clean --all all"
   echo $msg $cmd
   eval $cmd
}

macports-space-installed(){ macports-space installed ; }
macports-space-inactive(){  macports-space inactive ; }
macports-space(){
   local arg=${1:-installed}
   local space=space-$arg.txt
   [ ! -f $space ] && sudo port space $arg > $space
   grep MiB $space | sort -g -r | head -40
}



