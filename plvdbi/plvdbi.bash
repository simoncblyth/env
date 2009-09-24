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
#plvdbi-name(){  echo development ; }
plvdbi-name(){  echo deployment ; }
plvdbi-ini(){     
   local name=${1:-$(plvdbi-name)}
   echo $(plvdbi-dir)/$name.ini ;
 }
plvdbi-cd(){      cd $(plvdbi-dir); }

plvdbi-workdir(){ echo /tmp/env/plvdbi/workdir ; }


plvdbi-serve(){
  local msg="=== $FUNCNAME :" 
  rum-
  local iwd=$PWD 
  local dir=$(plvdbi-workdir)
  mkdir -p $dir && cd $dir
  echo $msg serving $(plvdbi-ini) from $PWD with $(which paster)
  
  case $(plvdbi-name) in
     development) paster serve --reload $(plvdbi-ini) ;;
               *) paster serve          $(plvdbi-ini) ;;
  esac
  cd $iwd
}


plvdbi-conf(){
   private-
   cat << EOC
           email_to=$(private-val PLVDBI_EMAIL_TO) 
        smtp_server=$(private-val PLVDBI_SMTP_SERVER) 
               port=$(private-val PLVDBI_PORT) 
   error_email_from=$(private-val PLVDBI_ERROR_EMAIL_FROM)
EOC
}

plvdbi-make-config(){
   local msg="=== $FUNCNAME :"
   private-
   local ini=$(plvdbi-ini deployment)
   local cmd="paster make-config plvdbi $ini $(echo $(plvdbi-conf)) ; svn revert $ini "
   echo $msg "$cmd"
   eval $cmd
}


