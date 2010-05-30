# === func-gen- : scons/scons fgp scons/scons.bash fgn scons fgh scons
scons-src(){      echo scons/scons.bash ; }
scons-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scons-src)} ; }
scons-vi(){       vi $(scons-source) ; }
scons-env(){      elocal- ; }
scons-usage(){
  cat << EOU
     scons-src : $(scons-src)
     scons-dir : $(scons-dir)

   == installs ==

     port installed v1.2.0.r3842 onto G, mostly into /opt/local/lib/scons-1.2.0
         sudo port install scons

     yum  installed v1.2.0.r3842  onto C,C2,N
         sudo yum install scons
         
     ipkg installed v1.2.0.r3842 
         sudo ipkg install scons

    
     http://prdownloads.sourceforge.net/scons/scons-1.3.0.tar.gz

EOU
}

scons-pth(){
  local msg="=== $FUNCNAME : "
  local tmp=/tmp/env/$FUNCNAME/scons.pth && mkdir -p $(dirname $tmp)
  echo $(scons-dir) > $tmp
  echo $msg prepare $tmp to put scons-dir on sys path 
  cat $tmp
  local cmd="sudo cp $tmp $(python-site)/$(basename $tmp)"
  echo $msg $cmd
  eval $cmd 
}

scons-dir(){ 
   pkgr- 
   case $(pkgr-cmd) in 
       port) echo /opt/local/lib/scons-1.2.0 ;; 
          *) echo /tmp ;;
   esac
}
scons-cd(){  cd $(scons-dir); }
scons-mate(){ mate $(scons-dir) ; }
scons-get(){
   local dir=$(dirname $(scons-dir)) &&  mkdir -p $dir && cd $dir

}
