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

pldep-selinux(){
   apache-
   apache-chcon $(pl-srcdir)
}

pldep-eggcache-dir(){ echo /var/cache/pl ; }
pldep-eggcache(){
  apache-
  apache-eggcache $(pldep-eggcache-dir)
}

## non-embedded deployment with apache mod_scgi or mod_fastcgi ?  or lighttpd/nginx

pldep-socket(){    echo /tmp/$(pl-projname).sock ; }
pldep-protocol(){  echo ${PLDEP_PROTOCOL:-scgi} ;}

pldep-command(){ cat << EOC
$(which paster) serve -v $(pldep-confpath) --server-name=$(pldep-protocol)
EOC
}

## interactive config check 
pldep-run(){       
   local msg="=== $FUNCNAME :"
   cd $(pl-projdir) 
   local ini=$(pldep-confpath)
   [ ! -f "$ini" ] && echo $msg ABORT no ini $ini && return 1
   local cmd="$(pldep-command)"
   echo $msg \"$cmd\"
   eval $cmd
}

pldep-sv-(){    
   cat << EOC
[program:$(pl-projname)]
command=$(pldep-command)
redirect_stderr=true
autostart=true
EOC
}

## supervisor hookup 
pldep-sv(){  sv-;sv-add $FUNCNAME- $(pl-projname).ini ; }

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
server:main|port|$(local-port $(pl-projname))
EOC
}

pldep-cnf(){
   echo -n 
}

pldep-modwsgi(){
   pl-
   PL_PROJNAME=dbi PL_INI=$(plvdbi-ini) pl-wsgi
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



pldep-wsgi-(){  
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
application = loadapp('config:$(pl-ini)')

EOS
}


pldep-wsgi(){
  local msg="=== $FUNCNAME :"
  local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
  local tmp=$tmpd/$(pl-projname).wsgi

  [ -z "$VIRTUAL_ENV" ] && echo $msg abort not in virtual env ... && return 1

  echo $msg writing $tmp ... using VIRTUAL_ENV $VIRTUAL_ENV
  $FUNCNAME- > $tmp
  modwsgi-
  modwsgi-deploy $tmp
}



