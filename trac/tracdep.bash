# === func-gen- : trac/tracdep fgp trac/tracdep.bash fgn tracdep fgh trac
tracdep-src(){      echo trac/tracdep.bash ; }
tracdep-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tracdep-src)} ; }
tracdep-vi(){       vi $(tracdep-source) ; }
tracdep-env(){      elocal- ; }
tracdep-usage(){
  cat << EOU
     tracdep-src : $(tracdep-src)
     tracdep-dir : $(tracdep-dir)

     tracdep-lighttpd-fcgi-

     Hmm these work ...
        sudo -u lighttpd TRAC_ENV_PARENT_DIR="/var/scm/tracs/" /data1/env/local/env/trac/package/tractrac/trac-0.11/trac/web/fcgi_frontend.py 
        sudo -u lighttpd env -i TRAC_ENV_PARENT_DIR="/var/scm/tracs/" /data1/env/local/env/trac/package/tractrac/trac-0.11/trac/web/fcgi_frontend.py 

    But when run from lighttpd-start selinux kicks in and get the usual sequence of access denied,
    following the usual selinux jousting arrive at : tracdep-selinux

    curl http://belle7.nuu.edu.tw:8080/tracs/env/
         due to the host bracketing "localhost" does not work  


      trac with lighttpd/fastcgi seems subjectively faster that with apache/modpython 


EOU
}
tracdep-dir(){ echo $(local-base)/env/trac/trac-tracdep ; }
tracdep-cd(){  cd $(tracdep-dir); }
tracdep-mate(){ mate $(tracdep-dir) ; }

tracdep-port(){  echo 8080 ; }
tracdep-prefix(){ echo tracs ; }
tracdep-test(){
   local msg="=== $FUNCNAME :"
   local cmd="curl http://$(hostname):$(tracdep-port)/$(tracdep-prefix)/env/ "
   echo $msg $cmd
   eval $cmd
}

tracdep-python(){ which python ; }
tracdep-tracsdir(){ echo /var/scm/tracs ; }
tracdep-reposdir(){ echo /var/scm/repos ; }
tracdep-eggcachedir(){ echo $(lighttpd-rundir)/eggcache ; }

tracdep-lighttpd-fcgi(){

  local msg="=== $FUNCNAME :"
  lighttpd-
  local conf=$(lighttpd-confd)/$FUNCNAME.conf
  [ ! -d "$(dirname $conf)" ] && echo $msg ABORT there is no confd  && return 1
  local tmp=/tmp/env/trac/$(basename $conf) && mkdir -p $(dirname $tmp)

  $FUNCNAME-  > $tmp

  if [ -f "$conf" ]; then
     diff $conf $tmp
  else
     cat $tmp
  fi

  local cmd="sudo cp $tmp $conf"
  local ans
  read -p "$msg Proceed with : $cmd : enter YES to continue  " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 1
  eval $cmd

  echo $msg incorporate into lighttpd with : include \"conf.d/$FUNCNAME.conf\" 

} 

tracdep-fcgipath(){ 
  trac-
  tractrac-
  echo $(tractrac-dir)/trac/web/fcgi_frontend.py 
}
tracdep-flupdir(){
   python -c "import flup ; import os ; print os.path.dirname(os.path.dirname(flup.__file__)) "
}
tracdep-htdocsdir(){
   trac-
   tractrac-
   echo $(tractrac-dir)/trac/htdocs
}

tracdep-fcgi-test(){
   cat << EOC
TRAC_ENV_PARENT_DIR="$(tracdep-tracsdir)/" $(tracdep-fcgipath) 
EOC
}


tracdep-selinux(){
   trac-
   tractrac-
   lighttpd-

   type $FUNCNAME

   ## code

   sudo chcon -R -h -t httpd_sys_script_exec_t $(tracdep-fcgipath)
   sudo chcon -R -h -t httpd_sys_content_t  $(tractrac-dir)
   sudo chcon -R -h -t httpd_sys_content_t  $(tracdep-flupdir)

   ## runtime 

   if [ ! -d "$(tracdep-eggcachedir)" ]; then
      sudo mkdir -p $(tracdep-eggcachedir) 
   fi
   lighttpd-chown $(tracdep-eggcachedir)
   sudo chcon -R -h -t httpd_sys_content_t $(tracdep-eggcachedir)

   ## content 

   lighttpd-chown $(tracdep-tracsdir)
   sudo chcon -R -h -t httpd_sys_content_t $(tracdep-tracsdir)
   lighttpd-chown $(tracdep-reposdir)
   sudo chcon -R -h -t httpd_sys_content_t $(tracdep-reposdir)

}

tracdep-lighttpd-fcgi-(){ 
  trac-
  tractrac-
  lighttpd-
  cat << EOC
#
#  http://trac.edgewall.org/wiki/TracFastCgi
#  http://redmine.lighttpd.net/wiki/1/HowToSetupTrac
#
\$HTTP["host"] == "$(hostname)" {
    server.document-root = "$(lighttpd-docroot)/" 
    alias.url            = (
        "/$(tracdep-prefix)/chrome/common/" => "$(tracdep-htdocsdir)/",
    )

    # rewrite for multiple svn project
    url.rewrite-final    = (
        "^/$(tracdep-prefix)/[^/]+/chrome/common/(.*)" => "/$(tracdep-prefix)/chrome/common/\$1",
    )

    \$HTTP["url"] =~ "^/$(tracdep-prefix)/chrome/" {
        # no fastcgi
    } 
    else \$HTTP["url"] =~ "^/$(tracdep-prefix)" {
        fastcgi.debug = 1
        fastcgi.server    = (
            "/$(tracdep-prefix)" => (          # if trac_prefix is empty, use "/" 
                (
                    # options needed to have lighty spawn trac
                    "bin-path"        => "$(tracdep-python) $(tracdep-fcgipath)",
                    "min-procs"       => 1,
                    "max-procs"       => 1,
                    "bin-environment" => (
                        "TRAC_ENV_PARENT_DIR" => "$(tracdep-tracsdir)/",
                        "PYTHON_EGG_CACHE" => "$(tracdep-eggcachedir)",
                    ),

                    # options needed in all cases
                    "socket"          => "$(lighttpd-rundir)/trac.sock",
                    "check-local"     => "disable",

                    # optional
                    "disable-time"    => 1,

                    # needed if trac_prefix is empty; and you need >= 1.4.23
                    "fix-root-scriptname" => "enable",
                ),
            ),
        )
    } 
} 

EOC
}

