# === func-gen- : geant4/geometry/g4daeserver/g4daeserver fgp geant4/geometry/g4daeserver/g4daeserver.bash fgn g4daeserver fgh geant4/geometry/g4daeserver
g4daeserver-src(){      echo geant4/geometry/g4daeserver/g4daeserver.bash ; }
g4daeserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4daeserver-src)} ; }
g4daeserver-vi(){       vi $(g4daeserver-source) ; }
g4daeserver-env(){      elocal- ; }
g4daeserver-usage(){ cat << EOU

DAESERVER
=========

installs
----------

D : g4daeserver vpython
~~~~~~~~~~~~~~~~~~~~~~~~~

Check apache is running, start SCGI g4daeserver with::

    delta:~ blyth$ g4daeserver.sh 
    2014-09-22 20:50:28,558 __main__ INFO     /Users/blyth/env/geant4/geometry/g4daeserver/g4daeserver.py 127.0.0.1:8080 scgi
    2014-09-22 20:50:28,558 __main__ INFO     g4daeserver startup with webpy 0.37 
    2014-09-22 20:50:28,614 env.geant4.geometry.collada.idmap INFO     found 685 unique ids 
    2014-09-22 20:50:28,625 env.geant4.geometry.collada.daenode INFO     idmap exists /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap entries 12230 
    2014-09-22 20:50:30,330 env.geant4.geometry.collada.daenode INFO     index linking DAENode with boundgeom 12230 volumes 
    2014-09-22 20:50:30,381 env.geant4.geometry.collada.daenode INFO     linking DAENode with idmap 12230 identifiers 
    2014-09-22 20:50:30 : WSGIServer starting up


Check some urls:

* http://localhost/dae/tree/3154.html



requirements
---------------

#. python packages

   * webpy
   * numpy
   * collada, `pycollada-vi` : nominally needs py26, 
     but succeded to backport to system py25 on G (for use with system panda3d/Cg)

#. apache, with SCGI configured as below
#. g4daeserver.py webpy process running, start with::

    simon:~ blyth$ g4daeserver.py
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




g4daeserver
-----------

::

    delta:g4daeserver blyth$ g4daeserver-
    delta:g4daeserver blyth$ g4daeserver-vrun
    2014-09-22 20:01:24,730 __main__ INFO     /Users/blyth/env/geant4/geometry/g4daeserver/g4daeserver.py 127.0.0.1:8080 scgi
    2014-09-22 20:01:24,730 __main__ INFO     g4daeserver startup with webpy 0.37 
    2014-09-22 20:01:24,786 env.geant4.geometry.collada.idmap INFO     found 685 unique ids 
    2014-09-22 20:01:24,797 env.geant4.geometry.collada.daenode INFO     idmap exists /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap entries 12230 
    2014-09-22 20:01:26,552 env.geant4.geometry.collada.daenode INFO     index linking DAENode with boundgeom 12230 volumes 
    2014-09-22 20:01:26,603 env.geant4.geometry.collada.daenode INFO     linking DAENode with idmap 12230 identifiers 
    2014-09-22 20:01:26 : WSGIServer starting up
    2014-09-22 20:01:26,670 scgi-wsgi INFO     WSGIServer starting up
    2014-09-22 20:01:47 : Protocol error 'invalid netstring length'
    2014-09-22 20:01:47,073 scgi-wsgi ERROR    Protocol error 'invalid netstring length'



apache configuration for webpy and statics
------------------------------------------- 

* http://webpy.org/cookbook/staticfiles
* http://webpy.org/deployment


httpd.conf::

    ######## DAESERVER 
     
    SCGIMount /dae 127.0.0.1:8080
     
    Alias /dae/static/ /Users/blyth/env/geant4/geometry/g4daeserver/static/
     
    <Directory /Users/blyth/env/geant4/geometry/g4daeserver/static>
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

    simon:g4daeserver blyth$ cp $(threejs-dir)/src/extras/helpers/AxisHelper.js static/r62/lib/
    simon:g4daeserver blyth$ cp $(threejs-dir)/src/extras/helpers/BoxHelper.js static/r62/lib/
    simon:g4daeserver blyth$ cp $(threejs-dir)/src/extras/helpers/ArrowHelper.js static/r62/lib/


EOU
}

g4daeserver-vdir(){ echo $(local-base)/env/geant4/geometry/g4daeserver_env ; }
g4daeserver-dir(){ echo $(env-home)/geant4/geometry/g4daeserver ; }
g4daeserver-cd(){  cd $(g4daeserver-dir); }
g4daeserver-mate(){ mate $(g4daeserver-dir) ; }
g4daeserver-get(){
   local dir=$(dirname $(g4daeserver-dir)) &&  mkdir -p $dir && cd $dir

}


g4daeserver-vrun(){
   export-
   export-export    # for DAE_NAME_DYB

   $(g4daeserver-vdir)/bin/python $(g4daeserver-dir)/g4daeserver.py $*
}


g4daeserver-apache-(){ cat << EOC

######## DAESERVER 
     
SCGIMount /dae 127.0.0.1:8080
     
Alias /dae/static/ $(env-home)/geant4/geometry/g4daeserver/static/
     
<Directory $(env-home)/geant4/geometry/g4daeserver/static>
Order deny,allow
Allow from all
</Directory>
     
#############################

EOC
}



g4daeserver-vinstall-deps(){
  
   g4daeserver--

   pycollada-
   pycollada-get      # gives error due to existing clone
   pycollada-build    # does very little as built already
   pycollada-install  # installs into the virtual python

   webpy-
   webpy-install

   

}

g4daeserver-v(){
    local vdir=$(g4daeserver-vdir)
    mkdir -p $(dirname $vdir)
    virtualenv --system-site-package $vdir
}
g4daeserver--(){
    source $(g4daeserver-vdir)/bin/activate
}


g4daeserver-start-args(){
  case $NODE_TAG in
      N) echo -w fcgi ;;
      G) echo -w scgi ;;  
      *) echo -n ;;
  esac
}

g4daeserver-start-cmd(){
  cat << EOC
$(which python) $(g4daeserver-dir)/g4daeserver.py $(g4daeserver-start-args) 
EOC
}

g4daeserver-log(){ echo $(local-base)/env/geant4/geometry/g4daeserver/logs/sv.log ; }


g4daeserver-sv-(){ 

mkdir -p $(dirname $(g4daeserver-log))
cat << EOX
[program:g4daeserver]
environment=LD_LIBRARY_PATH=$LD_LIBRARY_PATH,LOCAL_BASE=$LOCAL_BASE
command=$(g4daeserver-start-cmd)
process_name=%(program_name)s
autostart=true
autorestart=true

redirect_stderr=true
stdout_logfile=$(g4daeserver-log)
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=10


EOX
}
g4daeserver-sv(){
  sv- 
  $FUNCNAME- | sv-plus g4daeserver.ini
}


g4daeserver-libd(){ echo $(g4daeserver-dir)/static/r62/lib ; }
g4daeserver-collect(){
   threejs-
   g4daeserver-cd
   cp  $(threejs-dir)/examples/js/controls/TrackballControls.js $(g4daeserver-libd)/
   cp  $(threejs-dir)/examples/js/controls/OrbitControls.js $(g4daeserver-libd)/
}

