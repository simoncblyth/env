# === func-gen- : base/cpp.bash fgp base/cpp.bash fgn cpp
cpp-src(){      echo base/cpp.bash ; }
cpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpp-src)} ; }
cpp-vi(){       vi $(cpp-source) ; }
cpp-env(){      elocal- ; }
cpp-usage(){
  cat << EOU

C++
====



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



