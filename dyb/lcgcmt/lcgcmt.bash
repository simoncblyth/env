
lcgcmt-setup(){

   cd $SITEROOT/cmt/
   cmt config
   source setup.sh
   
}


lcgcmt-test(){

   cd ${SITEROOT}/lcgcmt/LCG_Settings/cmt
   cmt show macro LCG_system
   
}