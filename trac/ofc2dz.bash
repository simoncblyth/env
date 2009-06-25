# === func-gen- : trac/ofc2dz.bash fgp trac/ofc2dz.bash fgn ofc2dz
ofc2dz-src(){      echo trac/ofc2dz.bash ; }
ofc2dz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ofc2dz-src)} ; }
ofc2dz-vi(){       vi $(ofc2dz-source) ; }
ofc2dz-env(){      elocal- ; }
ofc2dz-usage(){
  cat << EOU
     ofc2dz-src : $(ofc2dz-src)

EOU
}

ofc2dz-cd(){  cd $(ofc2dz-dir) ; }
ofc2dz-dir(){ echo $(local-base)/env/ofc2dz ; }
ofc2dz-url(){ echo http://ofc2dz.com/OFC2/downloads/OFC2Patches-DZ-Ichor.zip ; }

ofc2dz-uzd(){ 
   local url=$(ofc2dz-url)
   local nam=$(basename $url)
   local uzd=${nam/.*}
   echo $uzd
}
ofc2dz-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(ofc2dz-dir)
   mkdir -p $dir && cd $dir
   local url=$(ofc2dz-url)
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -O $url
   local uzd=${nam/.*}
   echo $msg $nam $uzd 
   [ ! -d "$uzd" ] && unzip -d $uzd $nam
}
ofc2dz-pyget(){
  ofc2dz-cd
  git clone git://github.com/btbytes/pyofc2.git 
}
ofc2dz-swf(){
  echo $(ofc2dz-dir)/$(ofc2dz-uzd)/open-flash-chart/open-flash-chart.swf
}
ofc2dz-ln(){
  local swf=$(ofc2dz-swf)
  local cmd="sudo ln -s $swf $(apache-htdocs)/$(basename $swf)"
  echo $cmd
}


