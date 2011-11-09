

seed(){

   local iwd=$(pwd)
   cd $ENV_HOME/seed
   
   local src=seed.cc
   local exe=${CMTBIN}-seed
   [ $src -nt $exe ] && echo recompiling $src &&  g++ -o $exe $src 
   ./$exe

   cd $iwd
}
