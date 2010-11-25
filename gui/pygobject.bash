# === func-gen- : gui/pygobject fgp gui/pygobject.bash fgn pygobject fgh gui
pygobject-src(){      echo gui/pygobject.bash ; }
pygobject-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pygobject-src)} ; }
pygobject-vi(){       vi $(pygobject-source) ; }
pygobject-env(){      elocal- ; }
pygobject-usage(){
  cat << EOU
     pygobject-src : $(pygobject-src)
     pygobject-dir : $(pygobject-dir)

    http://live.gnome.org/PyGObject


    http://ftp.gnome.org/pub/GNOME/sources/pygobject/
    http://ftp.gnome.org/pub/GNOME/sources/pygobject/2.27/
         annoyingly older tarballs not available here ... 

     http://git.gnome.org/browse/pygobject
     http://git.gnome.org/browse/pygobject/tree/README


     checking for a Python interpreter with version >= 2.5.2... none
     configure: error: no suitable Python interpreter found

     [blyth@cms01 pygobject-2.27.0]$ which python
      /data/env/system/python/Python-2.5.1/bin/python

 

EOU
}

#pygobject-ver(){  echo 2.27 ; }   ## requires python_min_ver, 2.5.2
#pygobject-ver(){  echo 2.20 ; }    ## OK with 2.5.1... needs GLIB > 2.14
#pygobject-ver(){ echo 2.10 ; }     ## GLIB > 2.8.0
pygobject-ver(){ echo 2.8 ; }     ##  GLIB > 2.8.0


pygobject-name(){ echo pygobject-$(pygobject-ver).0 ; }

pygobject-dir(){ echo $(local-base)/env/gui/$(pygobject-name) ; }
pygobject-cd(){  cd $(pygobject-dir); }
pygobject-mate(){ mate $(pygobject-dir) ; }
pygobject-get(){
   local dir=$(dirname $(pygobject-dir)) &&  mkdir -p $dir && cd $dir
 
   local nam=$(pygobject-name)
   local tgz=$nam.tar.gz
   [ ! -f "$tgz" ] && curl -L -O http://ftp.gnome.org/pub/GNOME/sources/pygobject/$(pygobject-ver)/$tgz
   [ ! -d "$nam" ] && tar zxvf $tgz 
}

pygobject-git(){
   local dir=$(dirname $(pygobject-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://git.gnome.org/browse/pygobject
}


pygobject-configure(){
   pygobject-cd
   ./configure --prefix=$(python-site)
   
}


