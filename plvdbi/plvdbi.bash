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

     plvdbi-modscgi
         hints for apache proxying integration via scgi 

     plvdbi-shell
          
          gives error ... invalid literal for int() arising from :
          the response of /_test_vars not being covertible to integer 

             # Query the test app to setup the environment
             tresponse = test_app.get('/_test_vars')
             request_id = int(tresponse.body)

     plvdbi-archive-tw-resources
           collect the statics for deployment/serving from web server (apache/nginx/lighttpd) rather than webapp
               http://toscawidgets.org/documentation/ToscaWidgets/deploy.html
               http://projects.roggisch.de/tw/aggregation.html



EOU
}
plvdbi-dir(){     echo $(env-home)/plvdbi ; }
plvdbi-cd(){      cd $(plvdbi-dir); }
plvdbi-mate(){    mate $(plvdbi-dir) ; }
#plvdbi-name(){  echo development ; }
plvdbi-name(){  echo production ; }
plvdbi-ini(){     
   local name=${1:-$(plvdbi-name)}
   echo $(plvdbi-dir)/$name.ini ;
 }
plvdbi-edit(){    vim $(plvdbi-ini) ; }
plvdbi-workdir(){ echo /tmp/env/plvdbi/workdir ; }


plvdbi-build(){

    local msg="=== $FUNCNAME :"
    vdbi-
    vdbi-build
    [ ! $? -eq 0 ] && echo $msg ABORT after vdbi-build      && return 1  || echo $msg vdbi-build OK

    pl-
    pl-build 
    [ ! $? -eq 0 ] && echo $msg ABORT after pl-build        && return 1  || echo $msg pl-build OK

    authkit-
    authkit-build  
    [ ! $? -eq 0 ] && echo $msg ABORT after authkit-build   && return 1  || echo $msg authkit-build OK

    plvdbi-install 
    [ ! $? -eq 0 ] && echo $msg ABORT after -install && return 1  || echo $msg -install OK

    plvdbi-selinux  
    [ ! $? -eq 0 ] && echo $msg ABORT after -selinux && return 1  || echo $msg -selinux OK

    plvdbi-make-config 
    [ ! $? -eq 0 ] && return 1

    plvdbi-archive-tw-resources
    [ ! $? -eq 0 ] && return 1

}




plvdbi-setup(){
   plvdbi-cd
   python setup.py $*
}
plvdbi-install(){ plvdbi-setup develop ; }


plvdbi-selinux(){
   apache-
   apache-chcon $(plvdbi-dir)
}
  

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
   local ini=$(plvdbi-ini)
   local cmd="paster make-config plvdbi $ini $(echo $(plvdbi-conf)) ; svn revert $ini "
   echo $msg "$cmd"
   eval $cmd
}

plvdbi-shell(){
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   local iwd=$PWD
   cd $tmp
   paster --plugin=pylons shell $(plvdbi-ini)
   cd $iwd
}


plvdbi-statics-dir(){  echo $(plvdbi-dir)/plvdbi/public/toscawidgets ; }
plvdbi-archive-tw-resources(){
   cd $(plvdbi-dir)
   python setup.py archive_tw_resources  -f --output $(plvdbi-statics-dir)
}









