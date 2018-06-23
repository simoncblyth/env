/*

Right question::

    i got a third party library and one of the function takes an ifstream or
    ofstream for saving and reading data to/from a file. Consider that i can't
    modify that libray.. is there something i can do so i can get it to write to
    (or read from) a memory buffer instead of a physical file?  I want to do this
    because calling those function calls is the only way for me to retrieve the
    whole data stored in that object.. but writing/reading from disk slows down the
    process a lot!

Wrong approach at solution::

    If the functions really do take ofstream and ifstream references, and you
    really don't want to handle the I/O, then provided they don't call
    open/close/is_open, which are the only {o|i}fstream-specific functions, you can
    just get away with deriving new classes from std::ofstream and std::ifstream,
    which override all the protected methods, and forward them to a
    std::stringstream member.

    If the functions call the rdbuf member function, then you will also need to
    derive a class from std::filebuf, so you can return a pointer to it. This class
    will similarly need to forward all its members to the buffer of the contained
    stringstream


Right approach::

     stream redirection 

     http://wordaligned.org/articles/cpp-streambufs



*/


int main()
{
     return 0 ; 
}
