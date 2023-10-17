# === func-gen- : osx/mdls/mdls fgp osx/mdls/mdls.bash fgn mdls fgh osx/mdls
mdls-src(){      echo osx/mdls/mdls.bash ; }
mdls-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mdls-src)} ; }
mdls-vi(){       vi $(mdls-source) ; }
mdls-env(){      elocal- ; }
mdls-usage(){ cat << EOU



macos app programmatically swift access to mdls metadata

Programmatic access to mdls metadata ? ::

    MDItemRef item = MDItemCreateWithURL(NULL, (__bridge CFURLRef)url);
    NSArray* names = @[ (__bridge NSString*)kMDItemAlbum, /* ... */ ];
    NSDictionary* dictionary = CFBridgingRelease(MDItemCopyAttributes(item, (__bridge CFArrayRef)names));
    CFRelease(item);



* https://github.com/RhetTbull/osxmetadata

* https://developer.apple.com/documentation/foundation/nsurltagnameskey

* https://developer.apple.com/documentation/coreservices/file_metadata/mditem

EOU
}
mdls-dir(){ echo $(local-base)/env/osx/mdls/osx/mdls-mdls ; }
mdls-cd(){  cd $(mdls-dir); }

mdls-pngs(){
   local img
   ls -1 *.png | while read img 
   do 
       mdls-size $img
   done 
}

mdls-size(){
   local img=$1
   local width=$(mdls -name kMDItemPixelWidth -raw $img) 
   local height=$(mdls -name kMDItemPixelHeight -raw $img) 
   echo $img ${width}px_${height}px
}

