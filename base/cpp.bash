# === func-gen- : base/cpp.bash fgp base/cpp.bash fgn cpp
cpp-src(){      echo base/cpp.bash ; }
cpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpp-src)} ; }
cpp-vi(){       vi $(cpp-source) ; }
cpp-env(){      elocal- ; }
cpp-usage(){
  cat << EOU

C++
====


pragma once is non-standard
-----------------------------

* https://en.wikipedia.org/wiki/Pragma_once


thread-safety reentrant 
--------------------------

* https://www.ibm.com/support/knowledgecenter/en/ssw_aix_72/com.ibm.aix.genprogc/writing_reentrant_thread_safe_code.htm


cpp std::unique_ptr tis ok to return from functions
-----------------------------------------------------

* https://stackoverflow.com/questions/4316727/returning-unique-ptr-from-functions


cpp std::tie
--------------

* https://stackoverflow.com/questions/19800303/what-is-the-difference-between-assigning-to-stdtie-and-tuple-of-references

The goal of tie is making a temporary tuple to avoid temporary copies of tied
objects, the bad effect is, you can not return a tie if entry objects are local
temporary.

* http://blog.paphus.com/blog/2012/07/25/tuple-and-tie/



cpp streambufs : subverting a method that wants to write to a file to write to a buffer
-----------------------------------------------------------------------------------------

* https://stackoverflow.com/questions/8116541/what-exactly-is-streambuf-how-do-i-use-it

* http://wordaligned.org/articles/cpp-streambufs

::

    class redirecter
    // http://wordaligned.org/articles/cpp-streambufs
    {
    public:
        redirecter(std::ostream & dst, std::ostream & src)
            :   
            src(src), 
            sbuf(src.rdbuf(dst.rdbuf())) 
        {   
        }   

        ~redirecter() { src.rdbuf(sbuf); }
    private:
        std::ostream & src;
        std::streambuf * const sbuf;
    };

    void test_redirected( G4PhysicsOrderedFreeVector& vec, bool ascii )
    {
        std::ofstream fp("/dev/null", std::ios::out); 

        std::stringstream ss ;    
        redirecter rdir(ss,fp);    // redirect writes to the file to instead go to the stringstream 
        
        vec.Store(fp, ascii );

        std::cout <<  ss.str() << std::endl ; 
    }



tuple and tie
---------------

* http://blog.paphus.com/blog/2012/07/25/tuple-and-tie/

* *tie* provides convenient access to returned *tuple*

::

      #include <tuple>
      #include <iostream>

      using std::tuple;
      using std::tie;
      using std::make_tuple;
      using std::cout;

      class rectangle
      {
        public:
          rectangle(int _width, int _height) : width(_width), height(_height) {}
          int width, height;

          tuple<int, int> get_dimensions() {return make_tuple(width, height);}
      };

      int main(int argc, char** argv)
      {
          rectangle r(3,4);
          int w,h;
          tie(w,h) = r.get_dimensions();
          cout << w << ' ' << h << '\n';
          return 0;
      }



Inline Template Method Decl
-----------------------------

::

    template <typename T> void fill(T value) const ;


Template Method Impl
-----------------------

::

    template <typename T>
    void TBuf::fill(T value) const
    {


Forward Declare Template Class
-------------------------------

::

    template <typename T> class Constructor ;


Explicit instanciation of template class
------------------------------------------

::

    template class Constructor<OctreeNode> ;


Forward Instanciation of template method
-----------------------------------------

::

    template void TBuf::upload<float>(NPY<float>*) const ;


Instanciation visible across libs::

    270 template class THRAP_API TSparse<unsigned long long> ;
    271 template class THRAP_API TSparse<int> ;
    272 
    273 template THRAP_API void TSparse<unsigned long long>::apply_lookup<unsigned char>(CBufSlice target);
    274 template THRAP_API void TSparse<int>::apply_lookup<char>(CBufSlice target);



C++11
-------

* http://herbsutter.com/elements-of-modern-c-style/

Versions
---------


     cpp-src : $(cpp-src)

   hfag    "3.2.3 20030502 (Red Hat Linux 3.2.3-59)"
   grid1   "3.2.3 20030502 (Red Hat Linux 3.2.3-59)" 

   cms02   "3.4.6 20060404 (Red Hat 3.4.6-10)"
   cms01   "3.4.6 20060404 (Red Hat 3.4.6-11)"
   belle7  "4.1.2 20070626 (Red Hat 4.1.2-14)"

EOU
}
cpp-version(){ cpp-defines | grep __VERSION__ ; }
cpp-defines(){ echo | /usr/bin/c++ -x c++ -E -dD - ; }



