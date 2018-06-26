gio-source(){   echo ${BASH_SOURCE} ; }
gio-vi(){       vi $(gio-source) ; }
gio-env(){      elocal- ; }
gio-usage(){ cat << EOU

Groups.io Opticks
====================

* https://groups.io/g/opticks

* enable issues on bitbucket for installation debugging ? 
  to keep that out of the mailing list : aiming for 
  very low volume of mails 

Setup
-------

* DONE: make Opticks banner (900 x 300 pixels) for coverphoto


EOU
}


gio-open(){ open https://groups.io/g/opticks ; }
gio--(){ gio-open ; }
