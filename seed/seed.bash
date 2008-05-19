

seed(){

   local iwd=$(pwd)
   cd $ENV_HOME/seed
   
   local exe=${CMTBIN}-seed
   [ -f "$exe" ] || g++ -o $exe seed.cc
   ./$exe

   cd $iwd
}