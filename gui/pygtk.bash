# === func-gen- : gui/pygtk fgp gui/pygtk.bash fgn pygtk fgh gui
pygtk-src(){      echo gui/pygtk.bash ; }
pygtk-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pygtk-src)} ; }
pygtk-vi(){       vi $(pygtk-source) ; }
pygtk-env(){      elocal- ; }
pygtk-usage(){
  cat << EOU
     pygtk-src : $(pygtk-src)
     pygtk-dir : $(pygtk-dir)


     http://wiki.python.org/moin/GuiProgramming


     http://www.pygtk.org/downloads.html
     http://ftp.gnome.org/pub/GNOME/sources/pygtk/
     ftp://ftp.gtk.org/pub/gtk/python/README


     Matplotlib 1.0.0 (trunk-r8809) TkAgg shooting blanks on C (and N) 
         ... on N are using GTkAgg (easy as using system(yum) python )
         ... on C ... not so easy as source python

     [blyth@cms01 ~]$ pip install pygtk     ## pygtk-2.22.0.tar.bz2 
     [blyth@cms01 ~]$ pip install pygobject ## pygobject-2.26.0.tar.bz2    

         BUT THAT HANGS FOR INPUT AND THEN GIVES MESSAGE THAT NOT-SUPPORTED
         TO INSTALL VIA DISTUTILS

    Discerning the appropriate versions for C...

    == From N ... ==
        rpm -ql pygobject2
        ldd  /usr/lib/python2.4/site-packages/gtk-2.0/gobject/_gobject.so
        yum whatprovides /lib/libgobject-2.0.so.0

        yum info glib2      ==> 2.12.3
        yum info gtk2       ==> 2.10.4 
        yum info pygobject2 ==> 2.12.1
        yum info pygtk2     ==> 2.10.1
 
    == from pygtk-2.10.6 README ==

     47   * C compiler (GCC and MSVC supported)
     48   * Python 2.3.5 or higher
     49   * PyGObject 2.12.1 or higher
     50   * Glib 2.8.0 or higher
     51   * GTK+ 2.8.0 or higher (optional) or
     52     GTK+ 2.10.0 or higher for 2.10 API
     53   * libglade 2.5.0 or higher (optional)
     54   * pycairo 0.5.0 or higher (optional)
     55   * Numeric (optional)
     56 
     57 This release is supporting the following GTK+ releases:
     58 
     59   2.8.x
     60   2.10.x
     61 

  == pygtk 2.4.0 README ==

     42   * C compiler (GCC and MSVC supported)
     43   * Python 2.3 or higher
     44   * Glib 2.4 or higher
     45   * Gtk+ 2.4 or higher (optional)
     46   * libglade 2.3.6 (optional)
     47   * Numeric (optional)


    On C ..
        yum info glib2      ===>  2.4.7
        yum info gtk2        ===> 2.4.13
        yum info pygtk2      ===> 2.4.0        ( into system py2.3 )
        yum info pygobject ....  NOT FOUND      
             ( pre-pygobject era )


  == pygtk macports ==

    g4pb:site-packages blyth$ sudo port install py25-gtk 
    g4pb:site-packages blyth$ sudo port install py25-gtk 
       --->  Computing dependencies for py25-gtk
       --->  Dependencies to be installed: py25-cairo py25-numpy py25-gobject


    py25-numpy is dependency of py25-gtk ... liable to cause problems 




EOU
}
pygtk-dir(){ echo $(local-base)/env/gui/$(pygtk-name) ; }
pygtk-cd(){  cd $(pygtk-dir); }
pygtk-mate(){ mate $(pygtk-dir) ; }
pygtk-get(){
   local dir=$(dirname $(pygtk-dir)) &&  mkdir -p $dir && cd $dir

}
pygtk-ver(){  echo 2.4 ; }
pygtk-min(){  echo 0 ; }
pygtk-name(){ echo pygtk-$(pygtk-ver).$(pygtk-min) ; }

pygtk-url(){
   local ver=$(pygtk-ver)
   local nam=$(pygtk-name)
   case $ver in 
      2.4) echo ftp://ftp.gtk.org/pub/gtk/python/v$ver/$(pygtk-name).tar.gz ;;
        *) echo http://ftp.gnome.org/pub/GNOME/sources/pygtk/$ver/$(pygtk-name).tar.gz ;;
   esac
}


pygtk-get(){
   local dir=$(dirname $(pygtk-dir)) &&  mkdir -p $dir && cd $dir 
 
   local nam=$(pygtk-name)
   local tgz=$nam.tar.gz
   [ ! -f "$tgz" ] && curl -L -O $(pygtk-url)
   [ ! -d "$nam" ] && tar zxvf $tgz 
}


pygtk-configure(){
   pygtk-cd
   pythonbuild-
   ./configure --prefix=$(pythonbuild-prefix)
}

pygtk-build(){
   pygtk-cd
   make
   make install     ## i own the source python
}

pygtk-test(){
   local iwd=$PWD
   which python
   cd
   python -c "import gtk"
   cd $iwd
}

pygtk-installdir(){ python -c "import os, gtk as _ ; print os.path.dirname(_.__file__) " ; }
pygtk-version(){    python -c "import gtk as _ ; print _.pygtk_version  " ; }
pygtk-info(){  cat << EOI
    version      : $(pygtk-version)
    installdir   : $(pygtk-installdir)
EOI
}

