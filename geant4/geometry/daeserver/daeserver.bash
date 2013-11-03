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




EOU
}
daeserver-dir(){ echo $(env-home)/geant4/geometry/daeserver ; }
daeserver-cd(){  cd $(daeserver-dir); }
daeserver-mate(){ mate $(daeserver-dir) ; }
daeserver-get(){
   local dir=$(dirname $(daeserver-dir)) &&  mkdir -p $dir && cd $dir

}
