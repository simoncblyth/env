gio-source(){   echo ${BASH_SOURCE} ; }
gio-vi(){       vi $(gio-source) ; }
gio-env(){      elocal- ; }
gio-usage(){ cat << EOU

Groups.io Opticks
====================

* https://groups.io/g/opticks


TODO
------

* make logo for Opticks (900 x 300 pixels) for the coverphoto
* https://groups.io/g/opticks/coverphoto


EOU
}


gio-open(){ open https://groups.io/g/opticks ; }
gio--(){ gio-open ; }
