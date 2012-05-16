# === func-gen- : tools/svnlog fgp tools/svnlog.bash fgn svnlog fgh tools
svnlog-src(){      echo tools/svnlog.bash ; }
svnlog-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svnlog-src)} ; }
svnlog-vi(){       vi $(svnlog-source)  ; }
svnlog-env(){      elocal- ; }
svnlog-usage(){ cat << EOU

svnlog
=======

Queries SVN for commit messages 

Usage::
          
   svnlog-
   svnlog -h

   svnlog -a blyth     ## invoke from SVN working copy 


EOU
}
svnlog-dir(){ echo $(local-base)/env/tools/tools-svnlog ; }
svnlog-cd(){  cd $(svnlog-dir); }
svnlog-mate(){ mate $(svnlog-dir) ; }
svnlog(){
   python $(env-home)/tools/svnlog.py $*
}
