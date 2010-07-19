  meta          ,  class         ,  table       ,  CanL2Cache 
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