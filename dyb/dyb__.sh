
dyb__usage(){

 [ -z $BASH_SOURCE ] && echo oops : your bash lacks BASH_SOURCE ... its too old to play well with these functions 

cat << EOU


    Hookup the sourcing of this script into your environment with a function like :
    
      dyb_hookup(){
          local ddr=$1
          local dyb__=$ddr/dybgaudi/Utilities/Shell/bash/dyb__.sh 
          if [ -f $dyb__ ]; then
             . $dyb__
             [ -z $BASH_SOURCE ] && dyb__siteroot(){ echo $ddr ; }     ## workaround for older bash 
          fi 
      }     
      

         

    dyb__  <relpath> :  
         default $(dyb__default)
         
         jump into the cmt controlled environment and directory of a siteroot relative path, 
         where the siteroot is infered from the depth of this script 
         the relative path must either end with "cmt" or with a directory that 
         contains "cmt" 
       
         eg 
              dyb__ lcgcmt/LCG_Interfaces/ROOT/cmt  
              dyb__ lcgcmt/LCG_Interfaces/ROOT
              dyb__ dybgaudi/DybRelease

         NB the function "dyb" from dybgaudi/Utilities/Shell/bash/dyb.sh
            performs a similar task more cleverly, requiring a less explicit specification
            making it useful interactively 
            ... these functions however are more suited to automated usage 

    dyb__siteroot :  $(dyb__siteroot)
        siteroot obtained from the known depth of $BASH_SOURCE within the checkout
        unfortunately this needs a newer bash version to work ... as a workaround
        
        
           
  




    dyb__site  :
         invoke the site bootstrap setup.sh 

    dyb__test :
         test operation from a (nearly*) empty environment

             dyb__test "lcgcmt/LCG_Interfaces/ROOT   && env && which root "
             dyb__test "lcgcmt/LCG_Interfaces/Python && env && which python "

         (*) cmt setup.sh is sourcing .bash_profile forcing the inheriting of the original path as
             the sourceing may be due to  ${CMTROOT}/mgr/cmt  having a naked /bin/sh shebang line  

EOU
}



dyb__test(){
   [ -z $ORIGINAL_PATH ] && echo $FUNCNAME depends on ORIGINAL_PATH && return 1 
   env -i PATH=$ORIGINAL_PATH bash -noprofile -norc -c ". $BASH_SOURCE && dyb__ $* "
}



dyb__site(){

   local msg="=== $FUNCNAME :"
   
   unset SITEROOT
   unset CMTPROJECTPATH
   unset CMTPATH 
   unset CMTEXTRATAGS   

   local siteroot=$(dyb__siteroot)
   [ "$siteroot" == "." ] && echo $msg ERROR these functions must be defined from afar not from Utilities/bash folder && return 1
    
   [ ! -d $siteroot ] && echo $msg ERROR no siteroot $siteroot && return 1
   
   cd $siteroot

   [ ! -f setup.sh ] && echo $msg ERROR no $siteroot/setup.sh && return 1
   . setup.sh
}

dyb__siteroot(){
    echo $(dirname $(dirname $(dirname $(dirname $(dirname $BASH_SOURCE)))))
}

dyb__default(){
   echo dybgaudi/DybRelease
}


dyb__cmt(){
    [ ! -f "setup.sh" ] && cmt config
    [ ! -f "setup.sh" ] && echo $msg ERROR failed to create setup.sh && return 1
    . setup.sh   
}



dyb__(){   
    
	dyb__site
	 	
    local msg="=== $FUNCNAME :"        
    local rel=${1:-$(dyb__default)}
	local bas=$(basename $rel)
    
	# get rid of positional args to avoid a CMT warning 
    set --
	
    [ ! -d "$rel" ] && echo $msg ERROR argument rel $rel, no such directory && return 1 
    
    cd $rel
	
	if [ "$bas" == "cmt" ]; then
	  	dyb__cmt
    elif [ -d "cmt" ]; then
        cd cmt 
        dyb__cmt
        cd ..
    else
        echo $msg ERROR the relative path must end with the targeted cmt or its parent && return 1 
    fi
	
	# pwd
}


#dyb__ $*