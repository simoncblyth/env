# === func-gen- : base/sv fgp base/sv.bash fgn sv fgh base
sv-src(){      echo base/sv.bash ; }
sv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sv-src)} ; }
sv-vi(){       vi $(sv-source) ; }
sv-env(){      elocal- ; }
sv-usage(){
  cat << EOU
     sv-src : $(sv-src)
     sv-dir : $(sv-dir)

     sv-confpath : $(sv-confpath)

     sv-cnf-dump
     sv-cnf
         ConfigParser based dumping and get/get 
         (as ConfigObj does not handle ";" comments)

       http://supervisord.org/manual/current/


   Problems with config automation 
     1) ConfigObj doesnt handle ";" comments... and does not preserve spacing of inline # comments 
     2) Supervisor uses ConfigParser internally ... but this drops comments 
  
   Go with ConfigParser in ini-edit re-implementation ~/e/base/ini_cp.py   

EOU
}
sv-dir(){      echo $(local-base)/env/sv ; }
sv-confpath(){ echo $(sv-dir)/supervisord.conf ; }
sv-confdir(){  echo $(sv-dir)/conf ; }
sv-cd(){  cd $(sv-dir); }
sv-mate(){ mate $(sv-dir) ; }
sv-get(){
   python-
   local v=$(python-v)
   easy_install$v supervisor
   easy_install$v superlance    ## memmon + httpok 
}

sv-bootstrap(){
   local msg="=== $FUNCNAME :"
   local conf=$(sv-confpath)
   mkdir -p $(dirname $conf) 
   echo $msg writing to $conf
   echo_supervisord_conf  > $conf.sample   ## will be loosing the comments so write a sample to preserve them
   echo_supervisord_conf  > $conf
   mkdir -p $(sv-confdir)
   sv-cnf
}


sv-bins(){ echo supervisorctl supervisord echo_supervisord_conf pidproxy ; }
sv-check(){
   local msg="=== $FUNCNAME :"
   local bin ; for bin in $(sv-bins) ; do
     echo $msg $bin $(which $bin)
   done

}
sv-edit(){ vim $(sv-confpath) ; }
sv-start(){  supervisord   -c $(sv-confpath)    $* ; } 
sv-nstart(){ supervisord   -c $(sv-confpath) -n $* ; }   ## -n ... non daemon useful for debugging 
sv-ctl(){    supervisorctl -c $(sv-confpath) $* ; }

sv-add(){
   local msg="=== $FUNCNAME :"
   local fnc=$1
   local nam=$2
   local tmp=/tmp/env/$FUNCNAME/$nam && mkdir -p $(dirname $tmp)
   $fnc
   $fnc > $tmp
   local cmd="sudo cp $tmp $(sv-confdir)/ "
   echo $msg $cmd
   eval $cmd
}




sv-man-url(){ echo http://svn.supervisord.org/supervisor_manual/trunk ; }
sv-man-dir(){ echo $(sv-dir)/manual ; }
sv-man-cd(){  cd $(sv-man-dir) ; }
sv-man-get(){
   local dir=$(sv-man-dir) && mkdir -p $(dirname $dir) && cd $(dirname $dir)
   svn co $(sv-man-url) manual
}
sv-man-update(){
   local msg="=== $FUNCNAME :"
   [ "$(which xsltproc)" == "" ] && echo $msg ABORT no xsltproc && return 1
   sv-man-cd
   svn up
   autoconf   
   ./configure
   make
}
sv-man-open(){ open $(sv-man-dir)/html/index.html ; }




sv-dev-url(){ echo http://svn.supervisord.org/supervisor/trunk ; }
sv-dev-dir(){ echo $(sv-dir)/dev ; }
sv-dev-mate(){ mate $(sv-dev-dir) ; }
sv-dev-cd(){  cd $(sv-dev-dir) ; }
sv-dev-get(){
   local dir=$(sv-dev-dir) && mkdir -p $(dirname $dir) && cd $(dirname $dir)
   svn co $(sv-dev-url) dev
}
sv-dev-log(){ svn log $(sv-dev-dir) ;  }






sv-ini(){ sv-ini- $(sv-confpath) $*  ; }
sv-ini-() 
{ 
    local msg="=== $FUNCNAME :";
    local path=$1 ;
    shift;
    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp;
    local tpath=$tmp/$(basename $path);
    local cmd="cp $path $tpath ";
    eval $cmd;
    
    ## NB using a ConfigParser variant of the ConfigObj original as supervisor ini files
    ## use inline comments and semicolons that give ConfigParser indigestion
    
    INI_TRIPLET_DELIM="|" python $ENV_HOME/base/ini_cp.py $tpath $*;
    local dmd="diff $path $tpath";
    echo $msg $dmd;
    eval $dmd;
    [ "$?" == "0" ] && echo $msg no differences ... skipping && return 0;
    if [ -n "$SV_CONFIRM" ]; then
        local ans;
        read -p "$msg enter YES to confirm this change " ans;
        [ "$ans" != "YES" ] && echo $msg skipped && return 1;
    fi;
    $SUDO cp $tpath $path;
    [ "$user" != "$USER" ] && $SUDO chown $user:$user $path
}

sv-cnf-triplets-(){   cat << EOC

  include|files|conf/*.ini

  inet_http_server|port|127.0.0.1:9001 
  inet_http_server|user|realuser 
  inet_http_server|password|realpassword 

EOC
}
sv-cnf(){ sv-ini $(sv-cnf-triplets-);  }










sv-cfp-dump(){ $FUNCNAME- | python ; }
sv-cfp-dump-(){ cat << EOD
from ConfigParser import ConfigParser
c = ConfigParser()
c.read("$(sv-confpath)")
for section in c.sections():
    print section
    for option in c.options(section):
        print " ", option, "=", c.get(section, option)
EOD
}

sv-cfp(){  $FUNCNAME- | python - $* ; }
sv-cfp-(){ cat << EOD
## not used in anger ... see sv-ini
import sys
from ConfigParser import ConfigParser
c = ConfigParser()
c.read("$(sv-confpath)")
argv = sys.argv[1:]

if len(argv) == 0:
   c.write(sys.stdout)
elif len(argv) == 1:
   section = argv[0]
   for option in c.options(section):
       print " ", option, "=", c.get(section, option)
elif len(argv) == 2:
   print c.get(*argv)
elif len(argv) == 3:
   c.set(*argv)
   print "; $FUNCNAME set %s " %  repr(argv)
   c.write(sys.stdout)  
else:
   pass

EOD
}





