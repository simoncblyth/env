[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/graxml.bash

export GRAXML_NAME=GraXML-dist

export GRAXML_FOLDER=$LOCAL_BASE/graxml
export GRAXML_HOME=$GRAXML_FOLDER/GraXML

alias graxml="$GRAXML_FOLDER/GraXML/bin/GraXML.sh" 
alias graxmlconvertor="$GRAXML_FOLDER/GraXML/bin/GraXMLConvertor.sh" 


#
#  .gdml issues... 
#
#    1)   spaces between the attribute name and value, eg :
#                name= "hello"   are not appreciated by root
#
#
#
#  from the graxml beanshell....
#
#    w.show( "filepath.agdd")
#    calls
#       sg = new SceneGroup("Aberdeen_World.agdd");
#       w.show(sg)
#
#   with advantage of having a hook into sg...
#    (how to write a script to select volumes to show)
#
#  
#    System.out.println( sg.numChildren() );
#
#



gdml-correct(){
	gdml=$1
	echo args gdml:$gdml
	echo correcting the funny spaces before attribute values in gdml file that root does not like
    perl -pi -e 's/name=\s*/name=/g' $gdml

}

# root [0] TGeoManager::Import("Aberdeen_World.gdml")
# Info: TGeoManager::Import : Reading geometry from file: Aberdeen_World.gdml
# Error in <TXMLEngine::ParseFile>: Invalid attribute value, line 7
#
#  *** Break *** bus error
#
# Error: Unsupported GDML Tag Used :positionref. Please Check Geometry/Schema.
# Error: Unsupported GDML Tag Used :rotationref. Please Check Geometry/Schema.
#


#
# 635930 ERROR (GUI.RootWindow                : 612) : Failed to run script
# show.sh, see GraXML.log for details
#


#
#
#
#
#


graxml-test(){

   cd /tmp ; mkdir graxml-test ; cd graxml-test 
   cp $GRAXML_HOME/misc/Test/Solids.gdml .
   echo "w.show(\"Solids.gdml\");" > show.sh
   graxml show.sh
}



graxml-test-aberdeen(){

   n=graxml-test-aberdeen
   cd /tmp ; mkdir -p $n ; cd $n
   cp $DYW_FOLDER/viz/detectors/Aberdeen/Aberdeen_World_v1.gdml .

# _v1 is with the attributes corrected
   echo "w.show(\"Aberdeen_World_v1.gdml\");" > show.sh
   graxml show.sh
}




graxml-get(){

  n=$GRAXML_NAME
  cd $LOCAL_BASE
  test -d graxml || ( sudo mkdir graxml && sudo chown $user graxml )
  cd graxml
  
  #mkdir -p $n $n-build $n-src
  
  tgz=$n.tar.gz 
  url=http://hrivnac.web.cern.ch/hrivnac/Activities/Packages/$n.tar.gz


  #echo curl -o $tgz $url && tar zxvf $tgz 
  curl -o $tgz $url && tar zxvf $tgz 




}


