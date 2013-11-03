# === func-gen- : graphics/collada/meshtool fgp graphics/collada/meshtool.bash fgn meshtool fgh graphics/collada
meshtool-src(){      echo graphics/collada/meshtool.bash ; }
meshtool-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshtool-src)} ; }
meshtool-vi(){       vi $(meshtool-source) ; }
meshtool-env(){      elocal- ; }
meshtool-usage(){ cat << EOU

MESHTOOL
========

By the PyCollada author. Some visualisation options based on Panda3D. 
Also uses pycollada to do conversions.

Needs argparse, which is only standardly available from py26 so::

    sudo port install py26-argparse

BUT for use of panda3d, this needs to use the system python2.5.1
So::

    simon:~ blyth$ which pip    # promote /usr/local/bin into PATH
    /usr/local/bin/pip

    sudo pip install argparse 
 
SYSTEM PY2.5 INSTALLATION ATTEMPT
-------------------------------------

::

     simon:meshtool blyth$ sudo /usr/bin/python setup.py install

Loadsa python version issues from pycollada build attempt.
* /usr/local/env/graphics/collada/meshtool/system-py25-meshtool-log.txt

backport on py25compat branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

::

    simon:meshtool blyth$ git checkout -b py25compat
    Switched to a new branch 'py25compat'

    perl -pi -e 's,(except )(\S*)( as )(e:),$1$2\,$4,' meshtool/filters/__init__.py


ADD URL ARGUMENT SUPPORT
--------------------------

Support URL arguments::

    simon:meshtool blyth$ git diff filters/load_filters/load_collada.py
    diff --git a/meshtool/filters/load_filters/load_collada.py b/meshtool/filters/load_filters/load_collada.py
    index 651d5a6..2118648 100644
    --- a/meshtool/filters/load_filters/load_collada.py
    +++ b/meshtool/filters/load_filters/load_collada.py
    @@ -7,7 +7,10 @@ def FilterGenerator():
             def __init__(self):
                 super(ColladaLoadFilter, self).__init__('load_collada', 'Loads a collada file')
             def apply(self, filename):
    -            if not os.path.isfile(filename):
    +            if filename.startswith("http://"):
    +                import urllib2
    +                filename = urllib2.urlopen(filename)
    +            elif not os.path.isfile(filename):
                     raise FilterException("argument is not a valid file")
                 try:
                     col = collada.Collada(filename)
    @@ -18,4 +21,4 @@ def FilterGenerator():
                 return col
         return ColladaLoadFilter()
     from meshtool.filters import factory
    -factory.register(FilterGenerator().name, FilterGenerator)
    \ No newline at end of file
    +factory.register(FilterGenerator().name, FilterGenerator)
    simon:meshtool blyth$ 



EOU
}
meshtool-dir(){ echo $(local-base)/env/graphics/collada/meshtool ; }
meshtool-cd(){  cd $(meshtool-dir); }
meshtool-mate(){ mate $(meshtool-dir) ; }
meshtool-get(){
   local dir=$(dirname $(meshtool-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/pycollada/meshtool

}


meshtool(){ 
   type  $FUNCNAME
   export PRC_PATH=$HOME/.panda3d
   /usr/bin/python -c "from meshtool.__main__ import main ; main() " $* 
}
meshtool-view(){ 
   local msg="=== $FUNCNAME :"
   local arg=$1   
   local dae=${arg/.dae}.dae  # strip any preexisting .dae 
   local url=http://localhost:8080/geom/$dae
   collada-
   collada-cd
   if [ -f "$dae" ]; then 
      echo $msg use pre-existing $dae
      ls -l $dae
   else
      echo $msg downloading dae from url $url
      curl -sO $url
   fi
   local cmd="meshtool --load_collada $dae --viewer"
   echo $msg $cmd
   eval $cmd
}



