# === func-gen- : base/cpp.bash fgp base/cpp.bash fgn cpp
cpp-src(){      echo base/cpp.bash ; }
cpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpp-src)} ; }
cpp-vi(){       vi $(cpp-source) ; }
cpp-env(){      elocal- ; }
cpp-usage(){
  cat << EOU

C++
====


Sampling profilers
--------------------

* https://hackingcpp.com/cpp/tools/profilers.html
* https://github.com/jasonrohrer/wallClockProfiler


type convertor : eg root TComplex "decays" to real double
------------------------------------------------------------

See ~/env/root/UseTComplex.cc::

    struct TCheck
    {
        double fRe ; 
        double fIm ; 

        // TComplex has a convertor like this  
        operator double () const { return fRe; }
    };



pragma once is non-standard
-----------------------------

* https://en.wikipedia.org/wiki/Pragma_once

curated list of header only C++ libs
----------------------------------------

* https://github.com/p-ranav/awesome-hpp


command line parsers
---------------------

* https://github.com/p-ranav/argparse


IDE
-----

* https://gitlab.com/cppit/jucipp
* https://github.com/cppit/jucipp



ctor brackets
-----------------


* http://www.cplusplus.com/forum/beginner/54056/

If you have not provided a user-defined default constructor::

    Stock* a = new Stock();

will zero all the class members, whereas::

    Stock* a = new Stock;

will create an unitialized object.

As soon as you provide a constructor, the () no longer results in zero init.
You're responsible for init-ing all your own members.

Technically, the form with the bracket is not triggering the default
constructor. It's performing value initialization.

See "Is there an implicit default constructor in C++?" (esp. the final post)
http://stackoverflow.com/questions/563221/is-there-an-implicit-default-constructor-in-c

Andy


* http://stackoverflow.com/questions/563221/is-there-an-implicit-default-constructor-in-c

::

    new MyClass; // default-construction or no construction
    new MyClass(); // member-wise value-initialization

Note: If a user-declared default constructor does exist, then member-wise
value-initialization simply calls that and stops.


* this explains why with simple structs it is sometimes wise to 
  avoid defining any ctor because it simplifies copying : instead
  define an init method that has to be manually called 

* this explains why some Opticks nnode make a point of not defining ctors 



gcc programmatic backtrace
----------------------------

* https://panthema.net/2008/0901-stacktrace-demangled/
* https://panthema.net/2008/0901-stacktrace-demangled/backtrace.3.html

* https://stackoverflow.com/questions/36692315/what-exactly-does-rdynamic-do-and-when-exactly-is-it-needed

* https://github.com/melintea/Boost-Call_stack

* https://gcc.gnu.org/viewcvs/gcc/trunk/libbacktrace/


execinfo.h backtrace backtrace_symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.gnu.org/software/libc/manual/html_node/Backtraces.html

* https://stackoverflow.com/questions/8132913/objective-c-command-line-clang-print-stack-trace


See opticks/sysrap/SBacktrace.cc works : but not demangling properly 

* :google:`clang execinfo.h backtrace`

backtrace with file+linenos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/15129089/is-there-a-way-to-dump-stack-trace-with-line-number-from-a-linux-release-binary



lldb C++ API
--------------

* :google:`lldb/API/SB`

* http://lldb.llvm.org/cpp_reference/html/annotated.html
* http://llvm.org/svn/llvm-project/lldb/trunk/examples/lookup/main.cpp

* https://code.woboq.org/llvm/lldb/examples/functions/main.cpp.html
* https://github.com/cppit/jucipp/blob/924ccfc30da585c047b756813826a14efd5bc43e/src/debug.cc

* https://stackoverflow.com/questions/19019330/do-i-need-to-build-lldb-locally-to-use-c-interface

::

   epsilon:tmp blyth$ svn co http://llvm.org/svn/llvm-project/lldb/trunk/ lldb 


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



ordering critical when initializer lists call methods that populate vectors
-----------------------------------------------------------------------------

Spot the bug in the below::

    // hh

    struct Demo
    {
        unsigned initVec(); 

        //  the m_items vector should be before m_num_items 
        unsigned         m_num_items ; 
        std::vector<int> m_items ; 

    };


    // cc
    Demo::Demo()
       :
       m_num_items(initVec())
    { 
    }

    unsigned Demo::initVec()
    {
       m_items.push_back(1); 
       m_items.push_back(2); 
       m_items.push_back(3); 
       return m_items.size(); 
    }
 

Have observed the initVec appearing to work, but then being 
initialized back to being empty.  Solution is for the vector to 
be prior to m_num_items.



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



