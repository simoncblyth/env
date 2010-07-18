# === func-gen- : offline/gendbi fgp offline/gendbi.bash fgn gendbi fgh offline
gendbi-src(){      echo offline/gendbi.bash ; }
gendbi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gendbi-src)} ; }
gendbi-vi(){       vi $(gendbi-source) ; }
gendbi-srcdir(){   echo $(dirname $(gendbi-source)) ; }
gendbi-env(){      elocal- ; }
gendbi-usage(){
  cat << EOU
     gendbi-src : $(gendbi-src)
     gendbi-dir : $(gendbi-dir)


EOU
}
gendbi-dir(){ echo $(local-base)/env/offline/offline-gendbi ; }
gendbi-cd(){  cd $(gendbi-dir); }
gendbi-mate(){ mate $(gendbi-dir) ; }
gendbi-get(){
   local dir=$(dirname $(gendbi-dir)) &&  mkdir -p $dir && cd $dir

}

gendbi-csv-SimPmtSpec(){ cat << EOS
  meta          ,  class         ,  table
  1             ,  SimPmtSpec    ,  SimPmtSpec
; 
  name           , codetype                 , dbtype       , description 
  pmtId          , DayaBay::DetectorSensor  , int(11)      , PMT sensor ID 
  describ        , std::string              , varchar(27)  , String of decribing PMT position   
  gain           , double                   , float        , Relative gain for pmt with mean = 1 
  sigmaGain      , double                   , float        , 1-sigma spread of S.P.E. response
  timeOffset     , double                   , float        , Relative transit time offset
  timeSpread     , double                   , float        , Transit time spread
  efficiency     , double                   , float        , Absolute efficiency    
  prePulseProb   , double                   , float        , Probability of prepulsing
  afterPulseProb , double                   , float        , Probability of afterpulsing
  darkRate       , double                   , float        , Dark Rate
EOS
}


gendbi-csv1(){   cat << EOC
color,index,description
red,1,red devil
green,2,color of envy
blue,3,oceanic
cyan,4,ide
magenta,5,devine
yellow,6,ish ish
EOC
} 

gendbi-test(){
  gendbi-csv-SimPmtSpec | python $(gendbi-srcdir)/gendbi.py 

}





