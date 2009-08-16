#!/bin/bash

model=origin
particle=ibd

echo Model   :  $model
echo Particle:  $particle

echo Starting NuWa......
source $HOME/runNuWa.sh

cd $HOME/om_work/$model/$particle/

read -p "How many the files are?: " nu

for (( i=1; i<=$nu; i=i+1 ))
    do
    echo Analyzing $particle\_$i.root
    time nuwa.py -A None -n 1000 env.thho.NuWa.AVTest.chkIBD $particle\_$i.root > IbdBasicPlots_$i.log 2>&1
    mv -f IbdBasicPlots.root IbdBasicPlots_$i.root
    done


