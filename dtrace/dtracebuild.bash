
dtracebuild-usage(){

   cat << EOU

        dtracebuild-name : $(dtracebuild-name)
        dtracebuild-url  : $(dtracebuild-url)

        dtracebuild-get/configure/install


EOU

}

dtracebuild-env(){
  echo -n
}

dtracebuild-name(){ echo dtrace-20080803 ; }
dtracebuild-url(){  echo ftp://crisp.dynalias.com/pub/release/website/dtrace/$(dtracebuild-name).tar.bz2 ; }

dtracebuild-get(){

  echo -n



}


