
ABERDEEN_BASE=$ENV_BASE/aberdeen
export ABERDEEN_BASE

aberdeen_iwd=$(pwd)
cd $HOME/$ABERDEEN_BASE

roody-(){ [ -r $HOME/$ABERDEEN_BASE/roody.bash ] && . $HOME/$ABERDEEN_BASE/roody.bash ; }
midas-(){ [ -r $HOME/$ABERDEEN_BASE/midas.bash ] && . $HOME/$ABERDEEN_BASE/midas.bash ; }
rome-(){  [ -r $HOME/$ABERDEEN_BASE/rome.bash ]  && . $HOME/$ABERDEEN_BASE/rome.bash ; }
abd-(){  [ -r $HOME/$ABERDEEN_BASE/abd.bash ]  && . $HOME/$ABERDEEN_BASE/abd.bash ; }


roody-
rome-
abd-
midas-


roody-env
midas-path > /dev/null
roody-path > /dev/null

cd $aberdeen_iwd

