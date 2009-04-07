ezsetup-vi(){  vi $BASH_SOURCE ; }
ezsetup-env(){ elocal- ; }
ezsetup-usage(){

   cat << EOU

       ezsetup-get
             download and invoke ez_setup.py providing 
             setuptools module and easy_install entry point

EOU

}

ezsetup-tmp(){ echo /tmp/env/ezsetup ; }


ezsetup-get(){
    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local tmp=$(ezsetup-tmp) && mkdir -p $tmp
    cd $tmp

    local ezpy=ez_setup.py
    [ ! -f $ezpy ] && curl -L -O http://peak.telecommunity.com/dist/$ezpy
    local cmd="$(local-sudo) python $ezpy"
    echo $cmd ... from $PWD
    eval $cmd

    cd $iwd
}


