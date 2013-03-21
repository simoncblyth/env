# === func-gen- : tools/jinja2 fgp tools/jinja2.bash fgn jinja2 fgh tools
jinja2-src(){      echo tools/jinja2.bash ; }
jinja2-source(){   echo ${BASH_SOURCE:-$(env-home)/$(jinja2-src)} ; }
jinja2-vi(){       vi $(jinja2-source) ; }
jinja2-env(){      elocal- ; }
jinja2-usage(){ cat << EOU

sphinx-build is rather slow, suspect the jinja2 template 
filling is not using speedups and falling back to pure python

python environment on G somewhat confusing







simon:w blyth$ python -c "import markupsafe as _; print _.__file__ "
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/markupsafe/__init__.pyc

simon:w blyth$ python -c "import jinja2 as _ ; print _.__file__ "
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/jinja2/__init__.pyc



simon:w blyth$ /opt/local/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python -c  "import jinja2 as _ ; print _.__file__ "
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/jinja2/__init__.pyc

simon:w blyth$ /opt/local/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python -c  "import sphinx as _ ; print _.__file__ "
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/sphinx/__init__.pyc

simon:w blyth$ /opt/local/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python -c  "import markupsafe as _ ; print _.__file__ "
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/markupsafe/__init__.pyc


simon:w blyth$ port provides /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/markupsafe/__init__.pyc
Warning: port definitions are more than two weeks old, consider using selfupdate
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/markupsafe/__init__.pyc is not provided by a MacPorts port.
simon:w blyth$ 

simon:w blyth$ port provides /opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/markupsafe/__init__.py 
Warning: port definitions are more than two weeks old, consider using selfupdate
/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/markupsafe/__init__.py is not provided by a MacPorts port.
simon:w blyth$ 

   hmm maybe a python level install into the macports python


EOU
}
jinja2-dir(){ echo $(local-base)/env/tools/tools-jinja2 ; }
jinja2-cd(){  cd $(jinja2-dir); }
jinja2-mate(){ mate $(jinja2-dir) ; }
jinja2-get(){
   local dir=$(dirname $(jinja2-dir)) &&  mkdir -p $dir && cd $dir

}
