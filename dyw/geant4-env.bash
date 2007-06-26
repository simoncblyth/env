GQ_FOLDER=$LOCAL_BASE/geant4

if [ "X$GQ_NAME" == "X" ]; then
  #GQ_NAME="geant4.8.1.p01"
   GQ_NAME="geant4.8.2.p01"  
else
   echo honouring prior override GQ_NAME $GQ_NAME
fi


if [ "X$GQ_TAG" != "X" ]; then
   echo honouring prior GQ_TAG setting $GQ_TAG

elif [ "$NODE_TAG" == "L" ]; then   ## pal@nuu

   GQ_TAG=.

elif [ "$NODE_TAG" == "G" ]; then

   GQ_TAG=dbg
   
elif ([ "$NODE_TAG" == "P" ]) ; then

   #GQ_TAG="bat"    ##  no debug flags, no visualization ... for batch simulation runs
   GQ_TAG="dbg"     ##  debug flags + visualization
  

elif ([ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "$CLUSTER_TAG" ]) ; then

   #GQ_TAG="bat"    ##  no debug flags, no visualization ... for batch simulation runs
   GQ_TAG="dbg"     ##  debug flags + visualization
  
   
else   

   	GQ_TAG="dbg"

fi


export GQ_HOME=$GQ_FOLDER/$GQ_TAG/$GQ_NAME
export GQ_MACPATH=$GQ_FOLDER/macros:$HOME/geant4/macros
