setuptools-vi(){  vi $BASH_SOURCE ; }
setuptools-env(){ elocal- ; }
setuptools-usage(){

   cat << EOU


       setuptools-version : $(setuptools-version)

       setuptools-get
             download and invoke ez_setup.py providing 
             setuptools module and easy_install entry point

EOU

}

setuptools-tmp(){ echo /tmp/env/setuptools ; }
setuptools-version(){ python -c "import setuptools as _ ; print _.__version__" ; }
setuptools-get(){
    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local tmp=$(setuptools-tmp) && mkdir -p $tmp
    cd $tmp

    local ezpy=ez_setup.py
    [ ! -f $ezpy ] && curl -L -O http://peak.telecommunity.com/dist/$ezpy
    local cmd="$(local-sudo) python $ezpy"
    echo $cmd ... from $PWD
    eval $cmd

    cd $iwd
}


