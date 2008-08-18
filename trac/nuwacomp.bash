nuwacomp-usage(){
 
   cat << EOU

    THIS IS DEPRECATED BY PYTHON BASED SYNCING OF svn owner properties trac/autocomp/autocomp.bash 


      nuwacomp-list <dir-comntaing-dybgaudi-etc..>
            list of components to stdout, the svn property "owner" is examined to assign 
            an owner
            
                nuwacomp-list $SITEROOT >  nuwacomp.txt
                
                 SUDO=sudo traccomp-add nuwacomp.txt 
                
               
EOU


}


nuwacomp-env(){
   elocal-
   traccomp-
}

nuwacomp-list(){

  local base=${1:-$SITEROOT}

  cat << EOH

General /

Infrastructure / 
Infrastructure / Mailing Lists : cetull
Infrastructure / Official IHEP Web Site : mwang
Infrastructure / BNL Wikis : bv
Infrastructure / SVN : tianxc
Infrastructure / Trac : blyth

Installation /
Installation / Debian : bv
Installation / Mac OS X : dandwyer
Installation / Scientific Linux : patton

EOH
 
# 
#  for proj in $(nuwacomp-projs $base) ; do 
#     case $proj in 
#        relax|tutorial|lcgcmt|ldm) TRACCOMP_BRIEF=1 traccomp-from-wc $base/$proj ;; 
#                                *)                  traccomp-from-wc $base/$proj ;;
#     esac                       
#  done 
#
     
}


nuwacomp-projs(){
   local home=${1:-$SITEROOT}
   local iwd=$PWD
   cd $home
   local name
   for name in $(ls -1) ; do
      case $name in 
         Makefile|setup*|NuWa) echo -n ;;
                           *) echo $name ;;
      esac
   done
   cd $iwd

}

