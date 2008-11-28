#!/usr/bin/perl -w
#
#  createLibMacro.pl
#  Nathaniel Tagg, Dec 29 2006
#
#  This script simply changes the standard input of the form:
#   -lLib1 -LLib2
#  to a macro file of the form
#  {
#    gSystem->Load("libLib1");
#    gSystem->Load("libLib2");
#  }

$outfile=$ARGV[0];

$funcname = "";
if($outfile =~ "(.*).C") {  # The file is named *.C: take the '*'
  @temp=split("/",$1);
  $basename = pop(@temp);
  $funcname = "void $basename() ";  
}

print "Creating file $outfile\n";
open(OUTFILE,">$outfile");
print OUTFILE "//\n";
print OUTFILE "// Library loading macro created automatically.\n";
print OUTFILE "//\n";
print OUTFILE "\n";
print OUTFILE "$funcname\{\n";

$line=<STDIN>;
chomp $line;
@elements1=split(/ /,$line);
@elements=reverse @elements1;

$libpath="";
for $i (@elements) {
   if($i =~ "^-L(.*)") {
     $libpath = $libpath . ":$1"
   }
}

if($libpath ne "") {
   print OUTFILE "  TString s = gSystem->GetDynamicPath();\n";
   print OUTFILE "  s+=\"$libpath\";\n";
   print OUTFILE "  gSystem->SetDynamicPath(s.Data());\n";
}

for $i (@elements) {
   if($i =~ "^-l(.*)") {
     print OUTFILE "  gSystem->Load(\"lib$1\");\n";
   }   
}
print OUTFILE "}\n";

