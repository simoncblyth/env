# === func-gen- : offline/gendbi fgp offline/gendbi.bash fgn gendbi fgh offline
gendbi-src(){      echo offline/gendbi/gendbi.bash ; }
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
  meta          ,  class_        ,  table       ,  CanL2Cache 
  1             ,  SimPmtSpec    ,  SimPmtSpec  ,  kTRUE
; 
  name           , codetype                 , dbtype       , description                          , code2db 
  pmtId          , DayaBay::DetectorSensor  , int(11)      , PMT sensor ID                        , .sensorId()
  describ        , std::string              , varchar(27)  , String of decribing PMT position     ,
  gain           , double                   , float        , Relative gain for pmt with mean = 1  ,
  sigmaGain      , double                   , float        , 1-sigma spread of S.P.E. response    ,
  timeOffset     , double                   , float        , Relative transit time offset         ,
  timeSpread     , double                   , float        , Transit time spread                  ,
  efficiency     , double                   , float        , Absolute efficiency                  ,
  prePulseProb   , double                   , float        , Probability of prepulsing            ,
  afterPulseProb , double                   , float        , Probability of afterpulsing          ,
  darkRate       , double                   , float        , Dark Rate                            ,
EOS
}

gendbi-parse(){
  gendbi-csv-SimPmtSpec | python $(gendbi-srcdir)/parse.py 
}
gendbi-emit(){
  gendbi-csv-SimPmtSpec | python $(gendbi-srcdir)/emit.py 
}






