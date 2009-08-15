#!/bin/bash

model=origin
particle=gamma
energy=6mev

echo Model   :  $model
echo Particle:  $particle
echo Energy  :  $energy

echo Starting NuWa......
source $HOME/runNuWa.sh

cd $HOME/om_work/$model/$particle/$energy/

read -p "How many the files are?: " nu

for (( i=1; i<=$nu; i=i+1 ))
    do
    echo Analyzing $particle\_$energy\_$i.root
    time nuwa.py -A None -n 1000 env.thho.NuWa.AVTest.chkGamma $particle\_$energy\_$i.root > GammaBasicPlots_$i.log 2>&1
    mv GammaBasicPlots.root GammaBasicPlots_$i.root
    done


