# === func-gen- : git/gitorious fgp git/gitorious.bash fgn gitorious fgh git
gitorious-src(){      echo git/gitorious.bash ; }
gitorious-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitorious-src)} ; }
gitorious-vi(){       vi $(gitorious-source) ; }
gitorious-env(){      elocal- ; }
gitorious-usage(){
  cat << EOU
     gitorious-src : $(gitorious-src)
     gitorious-dir : $(gitorious-dir)

     http://gitorious.org/gitorious/mainline/blobs/master/README
     
     built on RoR ... looks difficult to install, many daemons
EOU
}
gitorious-dir(){ echo $(local-base)/env/git/git-gitorious ; }
gitorious-cd(){  cd $(gitorious-dir); }
gitorious-mate(){ mate $(gitorious-dir) ; }
gitorious-get(){
   #local dir=$(dirname $(gitorious-dir)) &&  mkdir -p $dir && cd $dir
   cd /tmp
   
   git clone git://gitorious.org/gitorious/mainline.git

}
