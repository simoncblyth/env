# === func-gen- : plvdbi/plvdbi fgp plvdbi/plvdbi.bash fgn plvdbi fgh plvdbi
plvdbi-src(){      echo plvdbi/plvdbi.bash ; }
plvdbi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(plvdbi-src)} ; }
plvdbi-vi(){       vi $(plvdbi-source) ; }

plvdbi-env(){      elocal- ; }
plvdbi-usage(){
  cat << EOU
     plvdbi-src : $(plvdbi-src)
     plvdbi-dir : $(plvdbi-dir)

     plvdbi-projdir : $(plvdbi-projdir)
     plvdbi-serve   
        run the server ... visible at http://localhost:5000 


EOU
}
plvdbi-dir(){     echo $(env-home)/plvdbi ; }
plvdbi-mate(){    mate $(plvdbi-dir) ; }
plvdbi-ini(){     echo $(plvdbi-dir)/development.ini ; }
plvdbi-cd(){      cd $(plvdbi-dir); }

plvdbi-workdir(){ echo /tmp/env/plvdbi/workdir ; }


plvdbi-serve(){
  local msg="=== $FUNCNAME :"
  rum-
  local iwd=$PWD 
  local dir=$(plvdbi-workdir)
  mkdir -p $dir && cd $dir
  echo $msg serving $(plvdbi-ini) from $PWD with $(which paster)
  paster serve --reload $(plvdbi-ini)

  cd $iwd
}


