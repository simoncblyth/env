DAESERVER
==========

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


apache configuration for webpy and statics
------------------------------------------- 

* http://webpy.org/cookbook/staticfiles
* http://webpy.org/deployment


httpd.conf::

    534 ######## DAESERVER 
    535 
    536 SCGIMount /dae 127.0.0.1:8080
    537 
    538 Alias /dae/static/ /Users/blyth/env/geant4/geometry/daeserver/static/
    539 
    540 <Directory /Users/blyth/env/geant4/geometry/daeserver/static>
    541 Order deny,allow
    542 Allow from all
    543 </Directory>
    544 
    545 #############################


webpy argv::

    if __name__ == "__main__":
        sys.argv[1:] = "127.0.0.1:8080 scgi".split()
        app = web.application(urls, globals())
        app.run()

Statics dont need webpy process running

* http://localhost/geo/static/hello.html

Dynamics do

* http://localhost/geo/?name=simon


gather pieces
---------------

::

    simon:WebGLBook blyth$ pwd
    /usr/local/env/graphics/webgl/WebGLBook

    simon:WebGLBook blyth$ cp -r sim ~/e/geant4/geometry/daeserver/static/
    simon:WebGLBook blyth$ cp -r css ~/e/geant4/geometry/daeserver/static/
    simon:WebGLBook blyth$ cp -r libs ~/e/geant4/geometry/daeserver/static/
    simon:WebGLBook blyth$ cp -r Chapter\ 7 ~/e/geant4/geometry/daeserver/static/   
    simon:WebGLBook blyth$ cp -r models/dybgeom ~/e/geant4/geometry/daeserver/static/models/ 

    http://localhost/geo/static/Chapter%207/production-loader-collada.html


::

    simon:static blyth$ cp Chapter\ 7/colladaModel.js .
    simon:static blyth$ cp Chapter\ 7/modelViewer.js .

