 [ "$DYW_DBG" == "1" ] && echo $DYW_BASE/dyw_build.bash

#  NB the order of the below, does matter 
 test -d $LOCAL_BASE || ( $SUDO mkdir -p $LOCAL_BASE ; $SUDO chown $USER $LOCAL_BASE )

 export ENV2GUI_VARLIST="LOCAL_NODE" 

 if [ "$NODE_TAG" != "N" ]; then 

 [ -r cmt.bash ]           && . cmt.bash

 [ -r clhep.bash ]         && . clhep.bash
 [ -r cernlib.bash ]       && . cernlib.bash 

 [ -r agdd.bash ]          && . agdd.bash

 [ -r coin3d.bash ]        && . coin3d.bash
 [ -r openmotif.bash ]     && . openmotif.bash
 [ -r soxt.bash ]          && . soxt.bash

 [ -r geant4.bash ]        && . geant4.bash

# comment out the below line, to have a G4 clean environment, as is usually needed when reconfiguring/rebuilding geant4 
 [ -r geant4_use.bash ]    && . geant4_use.bash  ##GEANT4_USE##

 [ -r dawn.bash ]          && . dawn.bash ##  
 [ -r graxml.bash ]        && . graxml.bash ##  
#[ -r gdml.bash ]          && . gdml.bash ## 

 [ -r root.bash ]          && . root.bash
 [ -r vgm.bash ]           && . vgm.bash
 [ -r xercesc.bash ]       && . xercesc.bash
 [ -r boost.bash ]         && . boost.bash

 [ -r aida.bash ]          && . aida.bash

 fi 


 [ -r dayabay.bash ]       && . dayabay.bash
 [ -r dayabay_extra.bash ] && . dayabay_extra.bash

# off while testing apache2
#[ -r apache.bash ]        && . apache.bash            ## webserver setup, to see autovalidation results 
 [ -r av.bash ]            && . av.bash                ## local autovalidation setup



###################################################







