#
#
#   Utilities for xml logging 
#
#
#   xml-fmtime
#   xml-stringlist-element
#   xml-file-element
#   xml-path-element
#   xml-content-element
#   xml-cdata-open
#   xml-cdata-close
#   xml-env-element
#   xml-ldd-element
#   xml-llp-element
#
#
#
#


xml-x(){ scp $HOME/$DYW_BASE/xml.bash ${1:-$TARGET_TAG}:$DYW_BASE; }

xml-fmtime(){
	perl -MPOSIX -e  "print strftime( '%Y%m%d-%H%M%S' , localtime($1) );" 
}

xml-path-element(){
	
  name=$1
  path=$2
  t=$(stat -c %Y $path) 
  printf "<path stamp=\"%s\" t=\"%d\" name=\"%s\" >%s</path>\n" $(xml-fmtime $t) $t $name $path

}

xml-stringlist-element(){
	
  name="$1"
  arglist="$2"
  
  printf "<%s>\n"  $name 
  printf "<line><![CDATA["   
  for arg in $arglist
  do
	 printf "%s " $arg  
  done	  
  printf "]]></line>\n"
  for arg in $arglist
  do
	 printf "<arg>%s</arg>\n" $arg  
  done	  
  printf "</%s>\n" $name
  
}

xml-file-element(){

  name=$1
  path=$2
  enam="file"

  printf "<%s>\n" $enam
  xml-path-element $name $path
  xml-content-element "content" $path  
  printf "</%s>\n" $enam

}


xml-content-element(){
  name=$1
  path=$2

  printf "<%s>\n" $name
  xml-cdata-open
  cat $path
  xml-cdata-close
  printf "</%s>\n" $name

}

xml-cdata-open(){
  printf "<![CDATA[\n" 
}

xml-cdata-close(){
  printf "]]>\n"
}


xml-env-element(){

  printf "<env>\n" 
  perl -e 'printf "<var name=\"%s\" >%s</var>\n", $_, $ENV{$_} for(sort keys %ENV)'
  printf "</env>\n" 
	
}


xml-ldd-element(){
 
  exe=$1
  printf "<ldd exe=\"%s\" >\n" $exe 
  libs=$(ldd $exe | perl -n -e 'm/^\s*(.*)\s=>\s(.*)\s\((.*)\).*$/ && print "$2 " ')
  for lib in $libs
  do
	  xml-path-element $(basename $lib) $lib 
  done	  
  printf "</ldd>\n" 
}



xml-llp-element(){
  
  printf "<llp>\n"
  echo $LD_LIBRARY_PATH | perl -lne 'printf "<lib>%s</lib>\n",$_ for(split(/:/))'
  printf "</llp>\n"
}



