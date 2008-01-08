#  
#    use the existing DYW machinery ... 
#
#       *   DYW_VERSION to legacy-aberdeen in .bash_profile
#       *   dyw-get   (which checks out and does dyw-localize on first use, updates thereafter)
#       *   cd $DYW ; dyw-build full    (NB the pwd is a "parameter" )
#
#
#
# abd-env(){
#
#  ABD_NAME=legacy-aberdeen
#  export ABD_NAME
#
#  ABD=$DYW_FOLDER/$ABD_NAME
#  export ABD 
#
# 
# }
#
#
#
#
# abd-update(){
#
#   abd-env
#   local abd_iwd=$(pwd)
#
#   cd $DYW_FOLDER
#   [ -d $ABD_NAME ] || svn co $DYBSVN/legacy/branches/$ABD_NAME
#   svn up $ABD_NAME
#   
#   cd $abd_iwd
#   
# }