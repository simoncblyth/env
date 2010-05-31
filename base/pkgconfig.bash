# === func-gen- : base/pkgconfig fgp base/pkgconfig.bash fgn pkgconfig fgh base
pkgconfig-src(){      echo base/pkgconfig.bash ; }
pkgconfig-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pkgconfig-src)} ; }
pkgconfig-vi(){       vi $(pkgconfig-source) ; }
pkgconfig-env(){      
    elocal- ; 
    export PKG_CONFIG_PATH=$(pkgconfig-dir)    ## hmm a bit brash 
}
pkgconfig-usage(){
  cat << EOU

     To understand... 
           man pkg-config

     PKG_CONFIG_PATH       : $PKG_CONFIG_PATH
     PKG_CONFIG_LIBDIR     : $PKG_CONFIG_LIBDIR

     pkgconfig-dir      :  $(pkgconfig-dir)
         directory in which to keep the .pc files

     pkgconfig-ls  
     pkgconfig-cd  
          ls/cd pkgconfig-dir

     pkgconfig-path pkg :  $(pkgconfig-path pkg)
         path of the pkg.pc file  

     pkgconfig-plus pkg 
       write what is piped into this function into pkg.pc file in pkgconfig directory  
       eg  
           rmqc- ; rmqc-pkgconfig- | pkgconfig-plus rmqc
       which is done by rmqc-pkgconfig
       

       After this can do :
          export PKG_CONFIG_PATH=$(pkgconfig-dir)     ## or could prepend to be less brash 
          pkg-config rmqc --cflags   
          pkg-config rmqc --libs
        
 
     == pkg-config background  ==

         pkg-config at prefix/bin/pkg-config  looks for .pc in prefix/lib/pkgconfig/

         For example the macports one :
                  /opt/local/bin/pkg-config
                  /opt/local/lib/pkgconfig/ 

         and the system one :
                  /usr/bin/pkg-config 
                  /usr/lib/pkgconfig/


EOU
}
pkgconfig-dir(){ echo $(local-base)/env/lib/pkgconfig ; }
pkgconfig-cd(){  cd $(pkgconfig-dir); }
pkgconfig-ls(){  ls -l $(pkgconfig-dir); }
pkgconfig-mate(){ mate $(pkgconfig-dir) ; }

pkgconfig-path(){ echo $(pkgconfig-dir)/${1:-error-no-pkg-name}.pc ; }
pkgconfig-plus(){
  local msg="=== $FUNCNAME :"
  local pkg=${1:-error-no-pkg-name}
  local tmp=/tmp/$USER/env/$FUNCNAME/$pkg.pc && mkdir -p $(dirname $tmp)
  echo $msg writing to $tmp
  cat - > $tmp
  cat $tmp
  local pcp=$(pkgconfig-path $pkg)
  local dir=$(dirname $pcp)
  [ ! -d "$dir" ] && echo $msg INFO creating $dir && mkdir -p $dir 
  if [ -f "$pcp" ]; then
     diff $pcp $tmp
  fi
  local cmd="cp $tmp $pcp "
  echo $msg $cmd 
  eval $cmd
}

pkgconfig--(){
  PKG_CONFIG_PATH=$(pkgconfig-dir) pkg-config $*
}
