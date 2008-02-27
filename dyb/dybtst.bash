

dybt-make(){

  test -d cmt && cd cmt && cmt config && . setup.sh && make || echo dybt-make FAILED from $PWD

}


dybt-helloworld(){

   cd $DDR/tutorial/HelloWorld
   
}


