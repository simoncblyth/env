

dyb-test-make(){

  test -d cmt && cd cmt && cmt config && . setup.sh && make || echo dyb-test-make FAILED from $PWD

}


dyb-test-helloworld(){

   cd $DDR/tutorial/HelloWorld
   
}


