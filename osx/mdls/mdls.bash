# === func-gen- : osx/mdls/mdls fgp osx/mdls/mdls.bash fgn mdls fgh osx/mdls
mdls-src(){      echo osx/mdls/mdls.bash ; }
mdls-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mdls-src)} ; }
mdls-vi(){       vi $(mdls-source) ; }
mdls-env(){      elocal- ; }
mdls-usage(){ cat << EOU





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

