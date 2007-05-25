

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



inversebeta-mac(){

# http://www.dayabay.caltech.edu/cgi-bin/twiki/bin/view/Main/PrimaryVertexGeneration
# 

hepevt=path-to-ascii-hepevt-file.txt

cat << EOM

/dyw/generator/select Generator2
/dyw/generator2/main HEP event
/dyw/generator2/hepevt $hepevt
# 
/dyw/generator2/pos glg4 2000 0 0 fill liquidscintillator

EOM
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


inversebeta-gen(){

  generator="inversebeta"
  dir=$DYW/Generators/InverseBeta/$CMTCONFIG
  exe=$dir/InverseBeta.exe
  gendir=$USER_BASE/dayabay/hepevt/$generator
  [ -d $gendir ] || ( echo WARNING creating $gendir && mkdir -p $gendir ) 
  
  nevts=${1:-100}
  seed=${2:-0}
  neutrino_angle_in_deg=${3:-0}
  
  hepevtfile=$gendir/inversebeta_${seed}_${neutrino_angle_in_deg}_${nevts}
  
  if [ -f $exe ]; then
     printf "<gen name=\"%s\" seed=\"%s\" neutrino_angle_in_deg=\"%s\" nevts=\"%s\" hepevtfile=\"%s\" >\n" $generator $seed $neutrino_angle_in_deg $nevts $hepevtfile
  else
     printf "<error> you need to build generator $exe first with inversebeta-build </error>"  && return 1
  fi    
  
  cd $dir
  $exe -h
  ##  InverseBeta.exe [-seed seed] [-o outputfilename] [-n nevents] [-angle neutrino_angle_in_deg] [-eplus_only] [-neutron_only] [-debug]
  $exe -seed $seed -o $hepevtfile -n $nevts -angle $neutrino_angle_in_deg 
  
  
  ##1:

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

}

