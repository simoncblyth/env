#!/bin/bash -l

generate(){
   local xml=com.nokia.Demo.xml
   local cls=DemoIf
   local nam=$(echo $cls | tr '[A-Z]' '[a-z]')
   local pcmd="qdbusxml2cpp -v -c $cls -p $nam.h:$nam.cpp $xml"
   echo $pcmd
   eval $pcmd
   local acmd="qdbusxml2cpp -v -c ${cls}Adaptor -a ${nam}adaptor.h:${nam}adaptor.cpp $xml"
   echo $acmd
   eval $acmd
}

generate $*

