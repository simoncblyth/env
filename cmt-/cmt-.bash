cmt-usage(){
   cat << EOU

    NB THIS IS FOR LIGHTWEIGHT FUNCTIONS ... WITHOUT STATE 



     cmt-cl  
        clean out *.sh *.csh Makefile

     cmt-i
        CMT envvars
    
     cmt--
        get into the CMT environment of the PWD, 
        or of a cmt subfolder if one exists


   CMT : it takes two to tangle  ... 2 heirarcies 
   ================================================
   
      projects  : linked up in cmt/project.cmt files
      packages  : linked up in cmt/requirements files 

ARE THESE TRUE ?
      packages depend on other packages and only refer to other packages
      projects depend on other projects and only refer to other projects


      CMTPROJECTPATH can contain only projects ... not packages directly 



EOU


}


cmt-env(){ echo -n ; }
cmt-cl(){  rm -f *.sh *.csh Makefile ; }
cmt-i(){   env | grep CMT ; }
cmt-wipe(){  
  local msg="=== $FUNCNAME :"
  local cnf=$CMTCONFIG 
  [ -z "$cnf" ] && echo $msg ERROR no cnf $cnf  && return 1
  [ ${#cnf} -lt 5 ] && echo $msg ERROR sanity check of cnf $cnf FAILED && return 1
 
  [ -d "$cnf" ]    && echo $msg deleting \"$cnf\"    && rm -rf "$cnf" 
  [ -d "../$cnf" ] && echo $msg deleting \"../$cnf\" && rm -rf "../$cnf" 
}
  


cmt--(){
   local msg="=== $FUNCNAME :"

   [ -d "cmt" ]      && echo $msg cd ==\> $PWD/cmt    &&  cd cmt
   [ -d "../cmt" ]   && echo $msg cd ==\> $PWD/../cmt &&  cd ../cmt
   [ $(basename $PWD) != "cmt" ] && echo $msg ERROR this only works from cmt package directories, or siblings/parents of such dirs && return 1 
   [ ! -f setup.sh ] && cmt config
   [ ! -f setup.sh ] && echo $msg ERROR failed to create setup.sh && return 1
   . setup.sh 
}


