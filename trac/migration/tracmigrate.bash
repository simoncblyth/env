# === func-gen- : trac/migration/tracmigrate fgp trac/migration/tracmigrate.bash fgn tracmigrate fgh trac/migration
tracmigrate-src(){      echo trac/migration/tracmigrate.bash ; }
tracmigrate-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tracmigrate-src)} ; }
tracmigrate-vi(){       vi $(tracmigrate-source) ; }
tracmigrate-env(){      elocal- ; }
tracmigrate-usage(){ cat << EOU

TRACMIGRATE
===========

Functionality is moved and generalized in scmmigrate-


EOU
}


