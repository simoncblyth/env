
[ "$DYW_DBG" == "1" ] && echo dyw_use.bash

dyw-use-x(){ scp $HOME/$DYW_BASE/dyw_use.bash ${1:-$TARGET_TAG}:$DYW_BASE ; }

#
# dont use home on grid1 as almost full disk
#   P dayabaysoft@grid1 
#   L sblyth@pal
#

##  removed ... this is set in env/base/local.bash 
## export SOURCE_NODE="g4pb"


if [ "$NODE_TAG" == "G" ]; then
   DYW_CVSROOT=":pserver:blyth@dayawane.ihep.ac.cn:/home/dybcvs/cvsroot" 
else
   DYW_CVSROOT=":pserver:dayabay@dayawane.ihep.ac.cn:/home/dybcvs/cvsroot" 
fi   

##
## editing access to repository only from source machine G,  read access otherwise
## 
export DYW_CVSROOT
export CVSROOT=$DYW_CVSROOT
##

#
##
##   NB definition of NODE_TAG CLUSTER_TAG LOCAL_BASE and USER_BASE moved to env/base/local.bash
##
## ------------- 

#export DYW_FOLDER_P=$LOCAL_BASE_P/dayabay
export DYW_FOLDER_P=$USER_BASE_P/dayabay

export DYW_FOLDER_L=$LOCAL_BASE_L/dayabay     ## formerly $HOME/Work/dayabay
export DYW_FOLDER_U=$LOCAL_BASE_U/dayabay    
export DYW_FOLDER_G=$USER_BASE_G/dayabay
export DYW_FOLDER_G1=$USER_BASE_G1/dayabay
export DYW_FOLDER_N=$USER_BASE_N/dayabay

vname=DYW_FOLDER_$NODE_TAG
eval _DYW_FOLDER=\$$vname
export DYW_FOLDER=${_DYW_FOLDER:-$DYW_FOLDER_U}

[ -d "$DYW_FOLDER" ] || ( echo WARNING creating DYW_FOLDER $DYW_FOLDER && mkdir -p $DYW_FOLDER )


## ------------  version string 

#export DYW_VERSION_G="dyw"                    ## historical current source with VGM enhancements + inverse_beta.cc enhancements   
 export DYW_VERSION_G="dyw_release_2_8_wc" 
 export DYW_VERSION_P="dyw_last_20070411"
 export DYW_VERSION_L="dyw_last_20070411"
 export DYW_VERSION_U="dyw_last_20070411"
#export DYW_VERSION_G1="dyw_20070503_wc"
 export DYW_VERSION_G1="dyw_release_2_8_wc"
 export DYW_VERSION_N="dyw_release_2_9_wc" 
 

 if [ "X$DYW_VERSION" == "X" ]; then
   vname=DYW_VERSION_$NODE_TAG
   eval DYW_VERSION=\$$vname
 else
   echo WARNING honouring a preset DYW_VERSION setting    
 fi
 export DYW_VERSION

## ------------  the local copies of the dyw cvs repository

#export DYW_P=${DYW_FOLDER_P}/${DYW_VERSION_P}
#export DYW_L=${DYW_FOLDER_L}/${DYW_VERSION_L}
#export DYW_U=${DYW_FOLDER_U}/${DYW_VERSION_U}
#export DYW_G=${DYW_FOLDER_G}/${DYW_VERSION_G}                 
#export DYW_G1=${DYW_FOLDER_G1}/${DYW_VERSION_G1}
#vname=DYW_$NODE_TAG
#eval DYW=\$$vname

export DYW=$DYW_FOLDER/$DYW_VERSION

## -------------- CMTPATH 

export CMTPATH=$DYW  ## inform cmt where to find the dayabay software, that was checked out of CVS

## ------------- macros   (currently mixed between user and admin ) 

export DYM_P=$LOCAL_BASE_P/dayabay/macros
export DYM_L=$LOCAL_BASE_L/dayabay/macros     ## formerly $HOME/Work/dayabay
export DYM_U=$LOCAL_BASE_U/dayabay/macros    

export DYM_G=$USER_BASE_G/dayabay/macros
export DYM_G1=$HOME/$ENV_BASE/macros    ## <<<<<< different behavior <<<<<<<<
export DYW_N=$HOME/$ENV_BASE/macros

vname=DYM_$NODE_TAG
eval _DYM=\$$vname
export DYM=${_DYM:-$DYM_U}


## autovalidation output folder 

#export DYW_AVOUT=$CMTPATH/AutoValidation/output	   
#export DYW_AVOUT=$DYW_FOLDER/autovalidation/output	   
export DYW_AVOUT=$USER_BASE/jobs/autovalidation
export DYW_AVCNF=$DYW_AVOUT/av_config.pl


 [ -r cmt_use.bash ]           && . cmt_use.bash
 [ -r xml.bash ]               && . xml.bash
 [ -r condor_use.bash ]        && . condor_use.bash
 [ -r condor_test.bash ]       && . condor_test.bash

 [ -r av_use.bash ]            && . av_use.bash
 [ -r g4dyb_use.bash ]         && . g4dyb_use.bash



dyw-use-macros(){
  cd $(dirname $DYM)
  test -d macros || mkdir macros 
  cd macros
  ls -alst 
}


dyw-use-env-info(){

	 printf "DYW_CVSROOT %-30s\n" $DYW_CVSROOT
	 printf "NODE_ABBREV %-30s\n" $NODE_ABBREV
	 printf "DYW_FOLDER  %-30s\n" $DYW_FOLDER
	 printf "DYW         %-30s\n" $DYW
	 printf "DYM         %-30s\n" $DYM
}



 
