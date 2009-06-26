autocomp-src(){    echo trac/autocomp/autocomp.bash ; }
autocomp-source(){ echo ${BASH_SOURCE:-$(env-home)/$(autocomp-src)} ; }
autocomp-dir(){    echo $(dirname $(autocomp-source)) ; }
autocomp-cd(){     cd $(autocomp-dir) ; }
autocomp-vi(){     vi $(autocomp-source) ; }

autocomp-usage(){

  cat << EOU

   Workflow :

       0) View the current Trac components with report:10 or {10} ...

            http://dayabay.ihep.ac.cn/tracs/dybsvn/report/10
            http://dayabay.phys.ntu.edu.tw/tracs/env/report/10
            http://localhost/tracs/workflow/report/10

       1) Modify the "owner" settings in instance functions
              env-owners-/workflow-owners-/...
          
          Apply these settings to relevant working copy with : 
              autocomp-owners
           
          Commit these property settings (and function changes) to the repository 


       2) On the machine on which the repository resides ... 
          (as need to access the trac and svn databases simultaneously)
          sync property settings with components :

              autocomp-sudosync





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
  





     trac-home        : $(trac-home)
            working copy home for TRAC_INSTANCE : $TRAC_INSTANCE
     
     autocomp-owners
            set the svn owner properties for paths relative to 
            trac-home, preexisting owner properties are overwritten

            the following folders / owners are defined :
          
$(autocomp-owners-)


            to manipulate an instance that is not the default for the node
            use the form :

                 TRAC_INSTANCE=env autocomp-usage
                 TRAC_INSTANCE=env autocomp-owners


  
EOU


}


autocomp-env(){
   trac-
}

autocomp-owners-set(){
   local msg="=== $FUNCNAME :"
   local fold=$1
   local owner=$2
   local path=$(trac-home)/$fold
   [ ! -d "$path" ] && echo $msg skip no existing dir $path && return 1
   ## preexisting owner gets overwritten
   local cmd="svn propset owner $owner $path "
   echo $cmd
   eval $cmd
}
autocomp-owners-(){ 
   ## defers to the function for the relevant instance 
   local name=${1:-$TRAC_INSTANCE}
   local f=${FUNCNAME/autocomp/$name}
   $f
}
autocomp-owners(){
   local msg="=== $FUNCNAME :"
   local name=${1:-$TRAC_INSTANCE}
   [ -z "$name" ] && echo $msg trac instance must be specified && return 1
   echo $msg set svn owner properties of folders relative to working copy home  $(trac-home)
   $FUNCNAME-
 
   local line
   $FUNCNAME- $name | while read line ; do 
      $FUNCNAME-set $line
   done
}



autocomp-py(){ vi $(autocomp-dir)/autocomponent.py ; }
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
