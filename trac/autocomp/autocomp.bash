autocomp-src(){    echo trac/autocomp/autocomp.bash ; }
autocomp-source(){ echo ${BASH_SOURCE:-$(env-home)/$(autocomp-src)} ; }
autocomp-dir(){    echo $(dirname $(autocomp-source)) ; }
autocomp-cd(){     cd $(autocomp-dir) ; }
autocomp-vi(){     vi $(autocomp-source) ; }

autocomp-usage(){

  cat << EOU


    View the components with {10} ...

            http://dayabay.ihep.ac.cn/tracs/dybsvn/report/10
            http://dayabay.phys.ntu.edu.tw/tracs/env/report/10
            http://localhost/tracs/workflow/report/10

 

     autocomp-sync  <env-name>    defaults to TRAC_INSTANCE 

             sync the components (names and owners) as specified by the owner
             properties on directories in the repository with the component list used for ticket creation 

            on cms01 when using direct $SUDO approacg run into
              /data/env/system/python/Python-2.5.1/bin/python: 
              error while loading shared libraries: libpython2.5.so.1.0: cannot open shared object file: No such file or directory
   
                 see #e111



    autocomp-sudosync  <env-name>  
              as the script writes to the trac log ... this sudo form is usally needed 


     autocomp-help           
     
             pydoc of the autocomponent module 
  
  
EOU


}


autocomp-env(){
   trac-
}


autocomp-sync(){
   
   local name=${1:-$TRAC_INSTANCE}
   sqlite-
   local cmd="python $(autocomp-dir)/autocomponent.py $(trac-envpath $name) $(trac-administrator)"
   echo $cmd
   eval $cmd
   
     
}


autocomp-sudosync(){
   sudo bash -lc "trac- ; autocomp- ; autocomp-sync $* "
}



autocomp-help(){

   local iwd=$PWD
   autocomp-cd 
   pydoc autocomponent
   
   cd $iwd
}
