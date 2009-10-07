pldep-src(){      echo pl/pldep.bash ; }
pldep-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pldep-src)} ; }
pldep-vi(){       vi $(pldep-source) ; }
pldep-env(){      elocal- ; }
pldep-usage(){
  cat << EOU

     NB operation of these commands depends on the basis parameters for the 
     specific app being defined previously, eg with plvdbi-

        pl-projname : $(pl-projname)
        pl-projdir  : $(pl-projdir)
        pl-confname : $(pl-confname)


     pldep-src : $(pldep-src)
     pldep-dir : $(pldep-dir)

     pldep-server : $(pldep-server) 
           eg scgi_threaded
           must correspond to <name> block in the deployment ini, server:<name>

     pldep-xcgi-run
          interactive test run of scgi/fcgi/afp running

     pldep-xcgi-sv
          add the non-embedded server to supervisor (sv-) control


     pldep-modwsgi
        prepare the modswgi script for apache embedded deployment

     pldep-lighttpd-
        incomplete config for lighttpd serving over fcgi 
 

EOU
}
pldep-dir(){ echo $(local-base)/env/pl/pl-pldep ; }
pldep-cd(){  cd $(pldep-dir); }
pldep-mate(){ mate $(pldep-dir) ; }
pldep-socket(){    echo /tmp/$(pl-projname).sock ; }
pldep-server(){  echo ${PLDEP_SERVER:-scgi_thread} ;}

pldep-selinux(){
   apache-
   apache-chcon $(pl-srcdir)
   pl-
   apache-chcon $(pl-projdir)
}

pldep-eggcache-dir(){ echo /var/cache/pl ; }
pldep-eggcache(){
  apache-
  apache-eggcache $(pldep-eggcache-dir)
}


## non-embedded deployment with apache mod_scgi or mod_fastcgi ?  or lighttpd/nginx

pldep-xcgi-(){ cat << EOC
$(which paster) serve -v $(pl-confpath) --server-name=$(pldep-server)
EOC
}

pldep-xcgi-run(){       
   local msg="=== $FUNCNAME :"
   cd $(pl-projdir) 
   local ini=$(pl-confpath)
   [ ! -f "$ini" ] && echo $msg ABORT no ini $ini && return 1
   local cmd="$(pldep-xcgi-)"
   echo $msg \"$cmd\"
   eval $cmd
}

pldep-xcgi-sv-(){    
   cat << EOC
[program:$(pl-projname)]
command=$(pldep-xcgi-)
redirect_stderr=true
autostart=true
EOC
}

pldep-xcgi-sv(){  sv-;sv-add $FUNCNAME- $(pl-projname).ini ; }







pldep-modwsgi-(){  
  python-
  cat << EOS

#  $FUNCNAME 
#     http://code.google.com/p/modwsgi/wiki/VirtualEnvironments

ALLDIRS = ["$VIRTUAL_ENV/lib/python$(python-major)/site-packages"]

import sys
import site

sys.stdout = sys.stderr


prev_sys_path = list(sys.path)

for dir in ALLDIRS:
    site.addsitedir(dir)

new_sys_path = []
for item in list(sys.path):
    if item not in prev_sys_path:
        new_sys_path.append(item)
        sys.path.remove(item)

sys.path[:0] = new_sys_path 

import os
os.environ['PYTHON_EGG_CACHE'] = '$(pldep-eggcache-dir)'
os.environ['ENV_PRIVATE_PATH'] = '$(apache-private-path)'

from paste.deploy import loadapp
application = loadapp('config:$(pl-confpath)')

EOS
}


pldep-modwsgi(){
  local msg="=== $FUNCNAME :"
  local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
  local tmp=$tmpd/$(pl-projname).wsgi

  [ -z "$VIRTUAL_ENV" ] && echo $msg abort not in virtual env ... && return 1

  echo $msg writing $tmp ... using VIRTUAL_ENV $VIRTUAL_ENV
  $FUNCNAME- > $tmp
  modwsgi-
  modwsgi-deploy $tmp
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


