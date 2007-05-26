

#
#  ========  dywGeneratorMessenger ===========
#
#   (generatorSelectCmd)    /dyw/generator/select
#                                                     dywGenerator->SetGeneratorType(newValue);
#   (genfileCmd)            /dyw/generator/genfile
#                                                      dywGenerator->SetFileName(newValue);
#   (generator2Cmd)        /dyw/generator2/main 
#                                                       dywGenerator->SetGenerator2MainCommand(newValue);
#                         choices:
#                            "GenericGun", "GenericGunRanMomentum", "HEP event"
#
#
#
#   (generator2PosCmd)     /dyw/generator2/pos
#    													dywGenerator->SetGenerator2PositionCommand(newValue);
#   (generator2HepEvtCmd)  /dyw/generator2/hepevt
#                                                       dywGenerator->SetGenerator2HepEvtCommand(newValue);
#
#                             The commands set the instance variables of the
#                             dywPrimaryGeneratorAction
#
#                                              glg4HepEventGenerator->SetState(
#
#
#         include/dywPrimaryGeneratorAction.hh 
#            src/dywGenerator2.cc
#



gen-lookup(){
   local generator=${1:-inversebeta}
   shift
   if [ "$generator" == "inversebeta" ]; then
       inversebeta-lookup $*
   else
       echo generator $generator not implemented in dyw_gen.bash
   fi    
}


gen-mac(){


 local generator=${1:-inversebeta}
 shift
 local  gentag=$(gen-lookup $generator gentag $*)
 local genfile=$(gen-lookup $generator genfile $*) 
 local   nevts=$(gen-lookup $generator nevts $*)

 local start_event_number=1
 local run_number=1
 local override_hostid=1 


cat << EOH
#  sourced from dyw_gen.bash::gen-mac
/files/output $gentag.root
/dyw/detector/select SingleModule
#
#
#  http://www.dayabay.caltech.edu/cgi-bin/twiki/bin/view/Main/EventReproducibility
#
#### for reproducibility of job.
## To reproduce one event, set the right random seed and generate the right primary particle and primary vertex. 
## To reproduce one run, set the right run number and host id.
#
/dyw/run/runNumber $run_number
/dyw/run/hostID $override_hostid
/dyw/event/startEventNumber $start_event_number
#
# this will override the three above settings
#/dyw/run/randomSeed $override_seed
#
EOH

mac-boilerplate

cat << EOG
#
# http://www.dayabay.caltech.edu/cgi-bin/twiki/bin/view/Main/PrimaryVertexGeneration
# 
/dyw/generator/select Generator2
/dyw/generator2/main HEP event
/dyw/generator2/hepevt $genfile
# 
/dyw/generator2/pos glg4 2000 0 0 fill liquidscintillator
#
EOG

echo "/run/beamOn $nevts "


}





mac-boilerplate(){

cat << EOB
#### Do NOT touch the following line.
/run/initialize 

####  Control the verbose
#     verbose level: 0: the least printout, 2: most detailed information.
/control/verbose 2
/run/verbose 2
/tracking/verbose 0
/dyw/phys/verbose 0 

####  Control the PMT Optical Model
#     verbose level :  0: quiet;
#                      1: minimal entrance/exit info;
#                      2: +print verbose tracking info;
#                      >=10:  lots of info on thin photocathode calcs.
#     luxlevel: 0: No transmitting and relfection on photocathode.
#               1: A simple model for transmitting and reflection.
#               >=2: Full PMT optical model. The default value is 3.
/PMTOpticalModel/verbose 0
/PMTOpticalModel/luxlevel 3

##### Physics process
/process/list
# /process/inactivate Cerenkov

####  Set scinitillation yield of the liquid Scintillator .
#     GdLS is for Gd-doped LS and LS is for normal LS in gamma catcher.
/dyw/detector/ScintYield_GdLS 9000.0 
/dyw/detector/ScintYield_LS 9000.0 


EOB

}


gen-run(){

 local generator=${1:-inversebeta}
 shift
 local gentag=$(gen-lookup $generator gentag $*)
 local genfile=$(gen-lookup $generator genfile $*) 

 if [ -f "$genfile" ]; then
   printf "<info> hepevtfile $file exists </info>" 
 else
   printf "<error> hepevtfile $file does not exist must $generator-gen first </error> " 
   return 1 
 fi
 
}




inversebeta-build(){

   dir=$DYW/Generators/InverseBeta/cmt
   if [ -d $dir ]; then 
     echo attempting to build/rebuild generator from $dir  
   else
     echo you need to install and set DYW $DYW first && return 1 
   fi 

   cd $dir
   cmt conf
   . setup.sh
   make

}


inversebeta-lookup(){

   local qwn=$1
   shift

   local generator="inversebeta"
   local nevts=${1:-100}
   local seed=${2:-0}
   local neutrino_angle_in_deg=${3:-0}
   
   local gentag=generator-${generator}_seed-${seed}_angle-${neutrino_angle_in_deg}_nevts-${nevts}
   local genfile=$gentag.txt
   local gendir=$USER_BASE/dayabay/hepevt/$generator
   local genxmlopen=$(printf "<gen name=\"%s\" seed=\"%s\" neutrino_angle_in_deg=\"%s\" nevts=\"%s\" gendir=\"%s\" gentag=\"%s\" file=\"%s\" >\n" $generator $seed $neutrino_angle_in_deg $nevts $gendir $gentag $file)
   local dir=$DYW/Generators/InverseBeta/$CMTCONFIG
   local exe=$dir/InverseBeta.exe
   local gencmd="cd $gendir ; $exe -h ; $exe -seed $seed -o $genfile -n $nevts -angle $neutrino_angle_in_deg "
   
   ##  InverseBeta.exe [-seed seed] [-o outputfilename] [-n nevents] [-angle neutrino_angle_in_deg] [-eplus_only] [-neutron_only] [-debug]
   
   eval val=\$$qwn
   echo $val
}


inversebeta-gen(){

   #  TODO:
   #      - timing 
   #      - date stamping
   #      - stamp run folder 

  local generator="inversebeta"
  local exe=$(inversebeta-lookup exe $*) 
  local gendir=$(inversebeta-lookup gendir $*)
  [ -d $gendir ] || ( printf "<warning> WARNING creating $gendir </warning>\n" && mkdir -p $gendir ) 
  
  local gentag=$(inversebeta-lookup gentag $*)
  local xmlopen=$(inversebeta-lookup genxmlopen $*)
  local cmd=$(inversebeta-lookup gencmd $*)
  
  echo $xmlopen     
  local error=""
  if [ -f "$exe" ]; then
     printf "<exe>%s</exe>\n" $exe
  else
     error="executable $exe doesnt exist , build it with $generator-build " 
  fi    
  
  ## without the quotes around $cmd gets whitespace split up 
  printf "<cmd>%s</cmd>\n"  "$cmd"
 
  if [ "X$error" == "X" ]; then 
     printf "<stdout>\n" 
     xml-cdata-open
     $cmd
     xml-cdata-close 
     printf "</stdout>\n"
  else
     printf "<error>%s</error>\n" $error
  fi
  
  printf "</gen>\n"
  
}

  
#   ============ looking at inversebeta.cc    ===================
#
#
# in debug mode 
#    canvas c1 , 3 plots   0..10
#
#        sigtot       rising
#        engspec      falling
#        totalprob    hump at 3.75 ish 
#
#    canvas c2 , 1 plot    0..10   "Positron Energy Spectrum"
#
#        mean energy    2.853   MeV    
#
#
#
#   default ... antineutrino.theta pi/2  
#                           .phi  -pi/2
#
#
#                  positron.theta peaks at pi/2  flat in .phi
#
#   nu_angle_wrt_y_in_deg   defaults to zero ... ie along y direction 
#
#
#
#         neutron.y      always -ve and less than -1.2 ish 
#         neutron.x/z    zero mean 
#
#         positron.x/y/z are roughly with zero mean
#
#         y
#         |
#         |
#         +----- x
#        /
#     z /
#
#
#
#    inversebeta->Draw("neutron.energy:neutron.y")
#


