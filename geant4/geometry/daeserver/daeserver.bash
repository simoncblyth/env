# === func-gen- : geant4/geometry/daeserver/daeserver fgp geant4/geometry/daeserver/daeserver.bash fgn daeserver fgh geant4/geometry/daeserver
daeserver-src(){      echo geant4/geometry/daeserver/daeserver.bash ; }
daeserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daeserver-src)} ; }
daeserver-vi(){       vi $(daeserver-source) ; }
daeserver-env(){      elocal- ; }
daeserver-usage(){ cat << EOU

DAESERVER
=========


requirements
---------------

#. python packages

   * webpy
   * numpy
   * collada, `pycollada-vi` : nominally needs py26, 
     but succeded to backport to system py25 on G (for use with system panda3d/Cg)

#. apache, with SCGI configured as below
#. daeserver.py webpy process running, start with::

    simon:~ blyth$ daeserver.py
    2013-11-02 15:07:37 : WSGIServer starting up
    2013-11-02 15:07:42 : GET /geo/hello/hello.html


#. test at http://localhost/dae/hello/hello.html?name=simon

nginx
------

* http://wiki.nginx.org/HttpScgiModule   This module first appeared in nginx-0.8.42

::

    nginx -v
    nginx version: nginx/0.6.39

::

    location /dae {
      scgi_pass localhost:8080;
    }


apache configuration for webpy and statics
------------------------------------------- 

* http://webpy.org/cookbook/staticfiles
* http://webpy.org/deployment


httpd.conf::

    ######## DAESERVER 
     
    SCGIMount /dae 127.0.0.1:8080
     
    Alias /dae/static/ /Users/blyth/env/geant4/geometry/daeserver/static/
     
    <Directory /Users/blyth/env/geant4/geometry/daeserver/static>
    Order deny,allow
    Allow from all
    </Directory>
     
    #############################


webpy argv::

    if __name__ == "__main__":
        sys.argv[1:] = "127.0.0.1:8080 scgi".split()
        app = web.application(urls, globals())
        app.run()

Statics dont need webpy process running

* http://localhost/geo/static/hello.html

Dynamics do

* http://localhost/geo/?name=simon



development sources
---------------------

::

    simon:daeserver blyth$ cp $(threejs-dir)/src/extras/helpers/AxisHelper.js static/r62/lib/
    simon:daeserver blyth$ cp $(threejs-dir)/src/extras/helpers/BoxHelper.js static/r62/lib/
    simon:daeserver blyth$ cp $(threejs-dir)/src/extras/helpers/ArrowHelper.js static/r62/lib/


EOU
}
daeserver-dir(){ echo $(env-home)/geant4/geometry/daeserver ; }
daeserver-cd(){  cd $(daeserver-dir); }
daeserver-mate(){ mate $(daeserver-dir) ; }
daeserver-get(){
   local dir=$(dirname $(daeserver-dir)) &&  mkdir -p $dir && cd $dir

}

daeserver-start-args(){
  case $NODE_TAG in
      N) echo -w fcgi ;;
      G) echo -w scgi ;;  
      *) echo -n ;;
  esac
}

daeserver-start-cmd(){
  cat << EOC
$(which python) $(daeserver-dir)/daeserver.py $(daeserver-start-args) 
EOC
}

daeserver-log(){ echo $(local-base)/env/geant4/geometry/daeserver/logs/sv.log ; }


daeserver-sv-(){ 

mkdir -p $(dirname $(daeserver-log))
cat << EOX
[program:daeserver]
environment=LD_LIBRARY_PATH=$LD_LIBRARY_PATH,LOCAL_BASE=$LOCAL_BASE
command=$(daeserver-start-cmd)
process_name=%(program_name)s
autostart=true
autorestart=true

redirect_stderr=true
stdout_logfile=$(daeserver-log)
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=10


EOX
}
daeserver-sv(){
  sv- 
  $FUNCNAME- | sv-plus daeserver.ini
}


daeserver-libd(){ echo $(daeserver-dir)/static/r62/lib ; }
daeserver-collect(){
   threejs-
   daeserver-cd
   cp  $(threejs-dir)/examples/js/controls/TrackballControls.js $(daeserver-libd)/
   cp  $(threejs-dir)/examples/js/controls/OrbitControls.js $(daeserver-libd)/
}

