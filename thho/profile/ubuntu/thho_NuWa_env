###################################################################
###################################################################
########### hook up the custom env for Dayabay SW #################
###################################################################
###################################################################

#################### The version of NUWA ##########################
# comment out the version you would like to use
#export NUWA_VERSION=1.0.0-rc01
###################################################################
nuwa_version=${NUWA_VERSION:-trunk}
echo The hooking up NuWa version is $nuwa_version now
export NUWA_HOME=/data/dyb/NuWa/$nuwa_version/NuWa-$nuwa_version
################## Blyth's env script ##########################
export ENV_HOME=$HOME/env
env-(){  . $ENV_HOME/env.bash  && env-env $* ; }
env-
###############################################################
nuwa-
################################################################
export DYB__OPEN_COMMAND=firefox
######### comment out the package you would like to use ###########
#export BUILD_PATH=dybgaudi/$nuwa_version/Simulation/GenTools
#export BUILD_PATH=dybgaudi/$nuwa_version/RootIO/RootIOTest
#export BUILD_PATH=dybgaudi/$nuwa_version/Validation/DetSimValidation
#export BUILD_PATH=dybgaudi/$nuwa_version/Simulation/ElecSim
export BUILD_PATH=tutorial/$nuwa_version/Simulation/SimHistsExample
#export BUILD_PATH=tutorial/$nuwa_version/DybHelloWorld
#export BUILD_PATH=dybgaudi/$nuwa_version/Simulation/ReadoutSim

#DYB_VERSION=trunk dyb_hookup /usr/local/dyb/trunk_dbg
#export DYB_CHECK=nope
