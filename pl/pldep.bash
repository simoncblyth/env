# === func-gen- : pl/pldep fgp pl/pldep.bash fgn pldep fgh pl
pldep-src(){      echo pl/pldep.bash ; }
pldep-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pldep-src)} ; }
pldep-vi(){       vi $(pldep-source) ; }
pldep-env(){      elocal- ; }
pldep-usage(){
  cat << EOU
     pldep-src : $(pldep-src)
     pldep-dir : $(pldep-dir)


EOU
}
pldep-dir(){ echo $(local-base)/env/pl/pl-pldep ; }
pldep-cd(){  cd $(pldep-dir); }
pldep-mate(){ mate $(pldep-dir) ; }
pldep-get(){
   local dir=$(dirname $(pldep-dir)) &&  mkdir -p $dir && cd $dir

}


## non-embedded deployment with apache mod_scgi or mod_fastcgi ?  or lighttpd/nginx

pldep-socket(){    echo /tmp/$(dj-project).sock ; }
pldep-protocol(){  echo scgi ;}
pldep-opts-fcgi(){ echo serve -v 2 debug=true protocol=fcgi socket=$(djdep-socket)  daemonize=false ; }
pldep-opts-scgi(){ echo serve -v 2 debug=true protocol=scgi host=$(modscgi-ip $(dj-project)) port=$(modscgi-port $(dj-project))  daemonize=false ; }


## interactive config check 
pldep-run(){       cd $(dj-projdir) ;  ./manage.py $(djdep-opts-$(djdep-protocol)) ;  }  

pldep-sv-(){    
   dj-
   cat << EOC
[program:$(dj-project)]
command=$(which paster) $(pldep-opts-$(pldep-protocol))
redirect_stderr=true
autostart=true
EOC
}

## supervisor hookup 
pldep-sv(){  
   sv-
   sv-add $FUNCNAME- $(dj-project).ini ; 
}

pldep-protocol(){ echo scgi ; }
pldep-protocol-egg(){
  case $(pldep-protocol) in 
    http) echo "egg:Paste#http" ;;
    scgi) echo "egg:Flup#scgi_thread" ;;
  esac     
}

pldep-cnf-triplets-(){
  modscgi-
cat << EOC
server:main|use|$(pldep-protocol-egg)
server:main|host|$(modscgi-ip)
server:main|port|$(local-port dbi)
EOC
}

pldep-cnf(){
  echo -n 
}



plvdbi-modwsgi(){
   pl-
   PL_PROJNAME=dbi PL_INI=$(plvdbi-ini) pl-wsgi
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

pldep-scgiroot(){  echo /plvdbi.scgi ; }
pldep-socket(){    echo /tmp/plvdbi.sock ; }
pldep-lighttpd-(){  cat << EOC

scgi.server = (
    "$(pldep-scgiroot)" => (
           "main" => (
               "socket" => "$(pldep-socket)",
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



