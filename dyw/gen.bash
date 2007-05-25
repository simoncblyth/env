

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
#         include/dywPrimaryGeneratorAction.hh 
#            src/dywGenerator2.cc
#




inversebeta-run(){

  dir=$DYW/Generators/InverseBeta/$CMTCONFIG
  exe=$fold/InverseBeta.exe
  [ -d $dir && -f $exe ] || ( echo you need to build generator $exe first && exit 1 )
  
  cd $dir
  $exe -h 

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

