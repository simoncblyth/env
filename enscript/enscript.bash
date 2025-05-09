

enscript-usage(){
 cat << EOU

   Installed to cut down on noise in the Trac log and 
   associated performance hit 
  
      enscript-name     :  $(enscript-name)
      enscript-url      :  $(enscript-url)
      enscript-dir      :  $(enscript-dir)
      enscript-builddir :  $(enscript-builddir)  
      
      enscript-cd/get/configure/make/install


  Installing from source as yum not cooperating ...
  


== message from the configure ==
   
Enscript is now configured to your system with the following
user-definable options.  Please, check that they are correct and
match to your system's properties.

Option     Change with configure's option   Current value
---------------------------------------------------------
Media      --with-media=MEDIA               A4
Spooler    --with-spooler=SPOOLER           lpr
PS level   --with-ps-level=LEVEL            2 
      


EOU

}

enscript-env(){
 elocal-
 enscript-path
}


enscript-path(){
  local bin=$(enscript-dir)/bin
  [ -d $bin ] && env-prepend $bin
}

enscript-test(){
  which enscript
  enscript --help
}


enscript-name(){
  echo enscript-1.6.1
}

enscript-url(){
  echo http://ftp.gnu.org/pub/gnu/enscript/$(enscript-name).tar.gz
}

enscript-dir(){
  echo  $(local-system-base)/enscript
}

enscript-builddir(){
  echo  $(local-system-base)/enscript/build/$(enscript-name)
}


enscript-cd(){
  cd $(enscript-dir)
}

enscript-get(){
   local iwd=$PWD
   local dir=$(enscript-dir)
   $SUDO mkdir -p $dir && cd $dir

   local nam=$(enscript-name)
   local tgz=$nam.tar.gz

   [ ! -f $tgz ] && curl -O $(enscript-url)
   
   mkdir -p build
   [ ! -d build/$nam ] && tar -C build -zxvf $tgz

   cd $iwd
}


enscript-configure(){
  cd $(enscript-builddir)
  ./configure --prefix=$(enscript-dir)  
 
}

enscript-make(){
  cd  $(enscript-builddir)
  make
}

enscript-install(){
  cd  $(enscript-builddir)
  $SUDO make install
}



