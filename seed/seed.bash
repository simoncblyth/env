

seed(){

   local iwd=$(pwd)
   cd $HOME/$ENV_BASE/seed
   
   g++ -o seed seed.cc
   ./seed

   cd $iwd
}