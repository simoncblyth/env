

seed(){

   local iwd=$(pwd)
   cd $HOME/$ENV_BASE/seed
   
   local exe=${CMTBIN}-seed
   [ -f "$exe" ] || g++ -o $exe seed.cc
   ./$exe

   cd $iwd
}