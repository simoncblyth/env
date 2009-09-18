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
plvdbi-dir(){ echo $(local-base)/env/plvdbi/plvdbi-plvdbi ; }
plvdbi-projdir(){ echo $(env-home)/plvdbi ; }
plvdbi-ini(){    echo $(plvdbi-projdir)/development.ini ; }

plvdbi-cd(){  cd $(plvdbi-dir); }
plvdbi-mate(){ mate $(plvdbi-dir) ; }
plvdbi-get(){
   local dir=$(dirname $(plvdbi-dir)) &&  mkdir -p $dir && cd $dir
}


plvdbi-serve(){
  local msg="=== $FUNCNAME :"
  rum-
  cd $(plvdbi-projdir) 
  echo $msg serving $(plvdbi-ini) from $PWD with $(which paster)
  paster serve --reload $(plvdbi-ini)
}


