
############## To setup the DayaBay utility bash scripts : ################
#  
#    Follow instructions in .bash_dyw to prepare the prerequisite exvironment
#
#############   To build the Dayabay "everything" from initial checkout :  ##############
#
#   quickstart :
#               1)  pick your repository , from the list at :
#                       http://hfag.phys.ntu.edu.tw:6060/tracs/
#                     eg if you pick  "dyw_release_2_5" then your tag is "release_2_5"
#            
#                2) checkout the repository using your tag , eg:
#
#                       dyw-checkout release_2_5
#
#                3)  set DYW_FOLDER appropiately for the working copy you just checked out:
#                    eg setting it to end with : dyw_release_2_5_wc 
#
#                       vi $HOME/env/base/local.bash 
#
#                4)  localize your requirements file
#                       dyw-requirements
#
#                5) localize your Geant4 requirements
#                        dyw-g4-req  
#                        cd $DYW ; svn commit  External/GEANT/cmt/requirements -m "localize "
#                   (commit is done in order to be able to run from a clean revision )    
#
#
#                  NB the version of Geant4 that you use 
#                     and type : 
#                          "dbg" with debug symbols + visualization support
#                          "bat" without those (factor of 3 faster)
#                     is fixed at this step based on the settings in 
#                       $HOME/env/dyw/geant4.bash
#
#                6) workaround grid1 time issues
#                       dyw-grid1-rootcint-timefix
#
#                7)   do a global cmt config   and build
#                        dyw-g4dyb-config
#
#                 8)  configure svn to ignore certain files : 
#                        svn-global-ignores
#                     thence 
#                        cd $DYW ; svn status -u   
#                     provides a short enough list to be useful
#            
#
#                  
#                       
#
#
#   usage :
#
#
#       dyw-checkout [tag]    checkout from the declared SVN repository 
#
#                             [tag] defaults to "release_2_5" in which case a repository called:
#                                   dyw_release_2_5 
#                             is checked out into folder 
#                                   $DYW_FOLDER/dyw_release_2_5_wc  (wc:working copy)
#
#                             for [tag] of head a symbolic link $DYW_FOLDER/dyw_head is read 
#                             to determine the name of the last head eg "dyw_20070503" 
#                             which is checked out into a new folder :
#                                   $DYW_FOLDER/dyw_20070503_wc
#
#
#                             usage examples :
#                                   dyw-checkout head
#                                   dyw-checkout 20070503
#                                   dyw-checkout release_2_5
#
#
#
#       dyw-get [tag]         CVS "login" and "get" into folder beneath $DYW_FOLDER and propagation to SVN
#
#                                  [tag] defaults to head, in which case the folder will be eg 
#                                       $DYW_FOLDER/dyw_20070503 
#                                       a symbolic link dyw_head to the newly created folder is made
#
#                               if [tag] is not "head" eg: release_2_5 then the folder will be 
#                                        $DYW_FOLDER/dyw_release_2_5 
#                                 and the CVS corresponding this tag will be "get"ed
#
#                               After the CVS get "scm-create" is invoked to propagate the pristine 
#                               and untouched CVS content into the SVN repository on target node, creating 
#                               a tracitory also 
#
#
#       dyw-update-deprecated   CVS update
#
#       dyw-env              dump the cmt env vars that will be used to construct the local requirements file
#
#       dyw-requirements     construct  the local requirements file
#
#       dyw-macros           list the macros , creates macros folder if not existing
#
#       dyw-req              list all requirements files
#       dyw-rf arg           search for arg in all requirements files 
#
#       dyw-g4-req            modify the geant4 requirements file "set"ings to match the
#                            geant4 data files available .. using geant4
#                            defaults overriding the dayabay ones
#
#       dyw                  cd to the heart of everything
#
#       dyw-config           do global CMT config of everything
#
#       dyw-cc [string]      list *string*.cc files beneath $DYW
#       dyw-h  [string]      list *string*.h files beneath $DYW
#       dyw-hh [string]      list *string*.hh files beneath $DYW
#
#       dyw-bmake            CMT broadcast make of everything
#
#
#       dyw-gen-on          switch on/off "use Generators" in Everything/cmt/requirements
#       dyw-gen-off
#
#       dyw-vis-on          modify the g4 requirements for visualization...  changing cppflags 
#       dyw-vis-off 
#
#       _mcevent            clean and make DataStructure/MCEvent 
#
#       ___g4dyb            config, setup, clean and make G4dyb ... use after external requirements changes
#       __g4dyb             config, setup and make G4dyb ... use after G4dyb requirements changes
#       _g4dyb              setup and make G4dyb ... quick make for use after G4dyb package code changes only 
#
#       g4dyb [macroname]   run G4dybApp.exe in batch mode, with single argument : macroname.mac 
#
#       g4dyb_s             run G4dybApp.exe without arguments, ie "session" mode
#       g4dyb_i [macroname] run G4dybApp.exe with arguments :  macroname.mac interactive
#                            verify your X11 setup first with  "xterm" , you might need to
#                            "ssh -Y G1" and/or  export DISPLAY="localhost:19.0" or some such   
#
#
#       g4dyb_              debug G4dybApp.exe by attaching gdb to the process
#       g4dyb__             direct executable gdb debugging 
#
#       g4dyb_env1          examining the CMT generated environment
#
#       g4dyb-momo          java interface experimentation
#
#
#
#
#   issues:
#
#        grid1 cernlib issues
#
#        without dyw-g4req get fatal error as G4 looking for a non-existing data file
#                  data/G4EMLOW3.0/rayl/re-ff-1.dat 
#
#
#        On Darwin comment out Generators in the G4dyb requirements, as no cernlib yet 
#         "dyw-gen-off" 
#
#        some cmt operations on Darwin... dont see my alias of gmake to make..  
#        hence create a link ...
#             [g4pb:/usr/bin] blyth$ sudo ln -s gnumake gmake
#
#
#        if /vis/open OGLIX  or /vis/open OIX results in "command not found" 
#        then probably the visualisation drivers are not setup
#        to confirm this, ...  at the start of job should see :
#
#        You have successfully registered the following graphics systems.
#		   [geant] =I= Current available graphics systems are:
#		                ASCIITree (ATree)
#						DAWNFILE (DAWNFILE)
#						GAGTree (GAGTree)
#						G4HepRep (HepRepXML)
#						G4HepRepFile (HepRepFile)
#						RayTracer (RayTracer)
#						VRML1FILE (VRML1FILE)
#						VRML2FILE (VRML2FILE)
#						OpenGLImmediateX (OGLIX)
#						OpenGLStoredX (OGLSX)
#						OpenInventorXt (OIX)
#																																									 
#         NB cannot just set the flags (G4VIS_USE) for compiling the main, in
#         G4dyb/cmt/requirements (as that will not link in all the appropriate libraries)... must apply
#         these flags to the External/Geant4/cmt/requirements 
#
#
#
#
#############  To cvs commit a modification into the repository  (do this from source machine (G) only )
#
#        dyw-login   (not needed ?)
#
#        check revision status of file intended  to be updated 
#
#           cvs log requirements       ## look at revision history
#           cvs diff requirements      ## compare local with the head version in the repository 
#
#           cvs commit -m "Add extra cppflag to allow creating shared libs on x86_64 architectures, such as AMD64 " requirements
#
#        to discard local changes and get a clean repository copy :
#
#           cvs update -C filename 
#
#        get help on a particular cvs command:
#            
#           cvs -H update 
#
#
#       NB these can be more easily done from the Xcode SCM GUI
#
##################################################################################################
#
#    cmt experience
#
#          doing a  ". cleanup.sh " unsets CMTPATH 
#
#
#
#





dayabay-i(){ [ -r $HOME/$DYW_BASE/dayabay.bash ] && . $HOME/$DYW_BASE/dayabay.bash ; }
dayabay-x(){ scp $HOME/$DYW_BASE/dayabay.bash ${1:-$TARGET_TAG}:$DYW_BASE; }



dyw-get-deprecated-following-move-to-svn(){  ## cvs login and initial get

  ## 
  ## instructions from: http://www.dayabay.caltech.edu/cgi-bin/twiki/bin/view/Main/DayaBayCVS
  ##
  ##      structure   dyw/dyw_$tag will avoid the issues
  ##
  ##  example tags are : 
  ##                    release_2_5
  ##
  ##  see the list at 
  ##                     http://www.dayabay.caltech.edu/cgi-bin/twiki/bin/view/Main/SoftwareReleases
  ##

  [ "$NODE_TAG" != "P" ] && echo "this is normally done from node P  dayabaysoft@grid1 " #&& return 1 
  ##[ "$NODE_TAG" != "G1" ] && echo "this is normally done from node G1  blyth@grid1 " && return 1
  ## actually it doesnt matter where this is done ... but clearer to use the same place each time

  local tag=${1:-HEAD}
  if [ "$tag" == "HEAD" ]; then
     dyw_tag=dyw_$(dyw-datestamp "now")
  else
     dyw_tag=dyw_$tag
  fi

  local name=$dyw_tag   ## used to be argument 2 
  local cvsroot=${2:-$DYW_CVSROOT_DAYABAY}   
  local cvspass=${3:-$DYW_PASS}

  cd $DYW_FOLDER

  if [ -d "$name" ]; then
      echo ==== dyw-get ======  a folder called $name exists already ... skipping checkout , updating instead
      cd $name
      cvs update -d
  else
      echo ==== dyw-get ====== proceeding to checkout from $cvsroot into $name ... tag $tag 
  
      mkdir ${name}
      if [ "$tag" == "HEAD" ]; then
         rm -f dyw_head && ln -s ${name} dyw_head 
      fi
      cd ${name}
      pwd
      
      $SCM_HOME/cvs-checkout.py $cvsroot $cvspass $tag
      
      #cvs -d $cvsroot login     ##  (once only ... it asks for CVS password ... the usual one worked )
      ## get the lot from the head or a particular tag
      #if [ "$tag" == "head" ]; then
      #   cvs -d $cvsroot get .            
      #else
      #   cvs -d $cvsroot get -r $tag .   
      #fi
      
          
              	  
  fi

  echo ==== dyw-get completed ====

  #echo =========== creating  remote repository called by the basename of the pwd , ie $dyw_tag with the contents of this pwd that was just got from CVS 
  #scm-create

}


dyw-login-deprecated(){
  cd $DYW
  cvs -d $DYW_CVSROOT login     
}


dyw-kludge(){
  
  prb=Generators/Muon/data/mountain_LA
  [ "$LOCAL_NODE" == "g4pb" ] && echo scp $DYW/$prb P:$DYW_P/$prb || echo do this from g4pb

}


dyw-update-deprecated(){   ## cvs update

  ## 
  ## note the value of CVSROOT doesnt matter for this...
  ## as the source is stored somewhere ... (in CVS folder ?)
  ## 
  ##  failing with Generators/Muon/data/mountain_LA
  ##  ... simple issue of no space on /home of grid1
  ##
  
  cd $DYW
  cvs update   

}











dyw-req(){   ## list requirements files
  cd $DYW 
  find . -name requirements  -exec ls -alst {} \;
}

dyw-rf(){   ## search for a string in the requirements file
  find $DYW/ -name 'requirements' -exec grep -H $1 {} \;
}

dyw(){     ## go to the heart of dyw cmt
  cd $DYW/Everything/cmt
}

dyw-cc(){  
	  cd $DYW && find . -name "*$1*.cc"  
}

dyw-hh(){  
	cd $DYW && find . -name "*$1*.hh"  
}

dyw-h(){   
	cd $DYW && find . -name "*$1*.h"  
}


dyw-everything-build(){ ##  do a global cmt config  

  cd $DYW/Everything/cmt/
   
  cmt broadcast clean

  cmt config                       ## creates setup.[c]sh and cleanup.[c]sh scripts in current directory 
  cmt broadcast echo hello      
  cmt broadcast cmt config         ## runs the command "cmt config" from all directories of all dependent packages 
  . setup.sh

  cmt broadcast make clean
  cmt broadcast make 



  
}






dyw-checkout(){  ## checkout from the declared SVN repository


  local t=${DYW_VERSION:4}   ## trim the dyw_  and _wc to get the tag string 
  t=${t%_wc}                   
  
  local branch=${1:-trunk}
  local tag=${2:-$t} 
   

   [ -d "$DYW_FOLDER" ] || mkdir -p $DYW_FOLDER 
   cd $DYW_FOLDER
 
  #
  # if specify head, then require to have the symbolic link to look up what daystamp
  # this corresponds to
  #
  if [ "$tag" == "HEAD" ]; then
     [ -L "dyw_head" ] || ( echo dyw_head must be a symbolic link && return 1 )
     dyw_tag=$(readlink dyw_head) 
  else
     dyw_tag=dyw_$tag
  fi

  dyw_wc=${dyw_tag}_wc
  [ -d "$dyw_wc" ] && echo a folder called $DYW_FOLDER/$dyw_wc exists already aborting dyw-checkout && return 1
 
  mkdir $dyw_wc
  cd $dyw_wc
 
  ## checkout 
  scm-checkout $dyw_tag $branch

}






dyw-branch-diff(){

   local b1="branches/blyth-optical"
   local b2="branches/thho-acrylic_sample"
   local ba="http://dayabay.phys.ntu.edu.tw/repos/dyw_release_2_9"
   
   local cmd1="svn diff $ba/$b1 $ba/$b2 $*"
   echo $cmd1
   eval $cmd1
   
   local cmd2="svn diff $ba/$b2 $ba/$b1 $*"
   echo $cmd2
   eval $cmd2

}




dyw-requirements-external(){

  pushd $DYW_SITE
  local req=requirements.$NODE_NAME
  if [ -f "$req" ]; then
    echo creating requirements link to $req
    rm -f requirements && ln -s $req requirements 
  else
    echo doing nothing as $req does not exist on NODE_NAME [$NODE_NAME]
  fi
  ls -alst
  
  popd   
}



dyw-diff-stamped(){

    ## recursive comparison of versions of the checked put respository,
	## ignoring temporary/derived files

    stamp=$(dyw-datestamp)
    cd $DYW_FOLDER
    diff -r --brief dyw dyw_${stamp} | perl -n -e '( m/CVS/ || m/setup/ || m/cleanup/ || m/\.make/ || m/\.DS_Store/ || m/Makefile/ || m/load\.C/ || m/Darwin/ || m/dyw\.1/ || m/xcodeproj/ ) || print $_  '  | sort 

}

dyw-datestamp(){

  if [ "$1" == "now" ]; then 
     if [ "$CMTCONFIG" == "Darwin" ] ; then
        timdef=$(perl -e 'print time')
	    refdef=$(date -r $timdef +"%Y%m%d")  
     else		
	    refdef=$(date  +"%Y%m%d")
     fi 
  else
     if [ "$LOCAL_NODE" == "grid1" ]; then
		 refdef="20070411" 
	 else
		 refdef="not-setup-yet" 
	 fi	 
  fi
  echo $refdef
}



dyw-build-deprecated(){


  local iwd=$PWD
  local defp="legacy/branches/$DYW_VERSION"
  local path=${1:-$defp}
  
  dyw-get $path
  local branch=$(basename $path)
  
  #
  # this parameter implies can just run on another branch, that has not been tested ..
  # ... set DYW_VERSION in .bash_profile is the standard approach 
  #
  
  echo ==== dyw-build building G4dybApp.exe from scratch with the latest from branch $branch
  cd $DYW_FOLDER
  
  
  if [ "X$branch" == "XHEAD" ]; then
  
     echo ===  dayabay user checkout/update of CVS repository HEAD
     dyw-get legacy/trunk 
     branch=$(basename $PWD) 
     
     echo === branch inferrred as $branch    
     dyw-localize $DYW_FOLDER/$branch   
         
           
  elif [ -d "$branch" ]; then
  
     cd $branch
     svn up 
      
  else
     svn co $dyw/branches/$branch
      
     cd $branch
     dyw-localize $PWD
 
     # remove the files that need to be localized , directly on the repository 
     # svn rm $dyw/branches/$branch/External/GEANT/cmt/requirements -m "removed as needs to be localized"
     # svn rm $dyw/branches/$branch/External/ROOT/cmt/fragments/rootcint -m "removed as needs to be localized"
     # thats too extreme !
     # for now live with committing them
        
     # have to commit and update in order to be at a clean svnversion   
     svn ci ./External/GEANT/cmt/requirements ./External/ROOT/cmt/fragments/rootcint -m "localized for node $NODE_TAG "
     svn up

  fi


  local orig_cmtpath=$CMTPATH
  CMTPATH=$DYW_FOLDER/$branch
  cd $CMTPATH/G4dyb/cmt
  
  echo ==== warning temporary reset of CMTPATH from $orig_cmtpath to $CMTPATH  === PWD $PWD ===
  
  [ -f requirements ] || ( echo ERROR error $PWD with the checkout/update && return 1 ) 

  local flags
  if [ "$GQ_TAG" == "dbg" ]; then
    flags="CMTEXTRATAGS=debug TMP=tmp"
  else
    flags="TMP=tmp" 
  fi  

  cmt br cmt config 
  cmt br make clean $flags
  cmt br make $flags

  CMTPATH=$orig_cmtpath  
  echo ==== resetting CMTPATH to $CMTPATH === PWD $PWD

  cd $iwd
  
}




