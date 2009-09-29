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

plvdbi-workdir(){ echo /tmp/env/plvdbi/workdir ; }


plvdbi-build(){


    vdbi-
    ! vdbi-build && return 1

    pl-
    ! pl-build   && return 1

    authkit-
    ! authkit-build   && return 1

    ! plvdbi-install  && return 1
    ! plvdbi-selinux  && return 1 


    ! plvdbi-make-config && return 1  
    ! plvdbi-archive-tw-resources  && return 1

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




##  daemon control 

plvdbi-start(){  plvdbi-pst start ; }
plvdbi-stop(){   plvdbi-pst stop ; }
plvdbi-reload(){ plvdbi-pst reload ; }
plvdbi-pst(){
   local msg="=== $FUNCNAME :"
   local arg=$1 
   case $arg in 
      start|stop|reload) echo $msg $arg ;;
                      *) echo "Usage plvdbi-pst start|stop|restart" && return 1 ;;
   esac    
   rum-
   local cmd="paster serve --daemon --pid-file=$(plvdbi-pid) --log-file=$(plvdbi-log)  $(plvdbi-ini) $arg"
   echo $msg $cmd
   eval $cmd
}
plvdbi-pid(){ echo  $(plvdbi-dir)/plvdbi.pid  ; }
plvdbi-log(){ echo  $(plvdbi-dir)/plvdbi.log  ; }
plvdbi-tail(){ tail -f $(plvdbi-log) ; }


plvdbi-modwsgi(){
   pl-
   PL_PROJNAME=dbi PL_INI=$(plvdbi-ini) pl-wsgi
}




## initd script

plvdbi-make-sh(){
  local sh=$(plvdbi-dir)/plvdbi.sh     ## for running in isolated situations 
  $FUNCNAME- > $sh
  #cat $sh 
  chmod u+x $sh
  #$sh hello
}

plvdbi-make-sh-(){   
   cat << EOS
#!/bin/bash 

pst(){ 
   local pid=$(plvdbi-pid)
   local log=$(plvdbi-log)
   local ini=$(plvdbi-ini)
   $(which paster) serve --daemon --pid-file=\$pid --log-file=\$log \$ini \$1 
}
#type pst
arg=\${1:-none}
case \$arg in
start|stop|restart) pst \$arg ;;
                 *) echo "Usage \$0 start|stop|restart "  ;; 
esac
exit 0
EOS

}


## lighttpd + scgi paste server ?

plvdbi-scgiroot(){  echo /plvdbi.scgi ; }
plvdbi-socket(){    echo /tmp/plvdbi.sock ; }
plvdbi-lighttpd-(){  cat << EOC

scgi.server = (
    "$(plvdbi-scgiroot)" => (
           "main" => (
               "socket" => "$(plvdbi-socket)",
               "check-local" => "disable",
               "allow-x-send-file" => "enable" , 
                      )
                 ),
)

# The alias module is used to specify a special document-root for a given url-subset. 
alias.url += (
           "/toscawidgets" => "$(plvdbi-dir)/plvdbi/public/toscawidgets",  
)

url.rewrite-once += (
      "^(/toscawidgets.*)$" => "\$1",
      "^/favicon\.ico$" => "/toscawidgets/favicon.ico",
      "^/robots\.txt$" => "/robots.txt",
      "^(/.*)$" => "$(plvdbi-scgiroot)\$1",
)

EOC
}







