

dybt-usage(){

cat << EOU

   scratch pad ...
   



EOU



}



dybt-env(){

   elocal-

}



dybt-make(){

  test -d cmt && cd cmt && cmt config && . setup.sh && make || echo dybt-make FAILED from $PWD

}


dybt-helloworld(){

   cd $DDR/tutorial/HelloWorld
   
}

dybt-py-import(){

  dybr-
  dybr-projs
  ipython -c "import $* "

  ##
  ##
  ##   dybt- ; dybt-py-import PyCintex   
  ##  
  ##   fails  "No module named libPyROOT "
  ##

}
