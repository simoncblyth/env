
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath> 
#include <cassert>


// sysrap-
#include "SDigest.hh"
// brap-
#include "BFile.hh"
#include "BMap.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

// ggeo-
#include "GProperty.hh"
#include "GDomain.hh"
#include "GAry.hh"

#include "PLOG.hh"



template <typename T>
const char* GProperty<T>::DOMAIN_FMT = " %10.3f" ; 

template <typename T>
const char* GProperty<T>::VALUE_FMT = " %10.3f" ; 




template <typename T>
T GProperty<T>::getValue(unsigned int index)
{
    return m_values->getValue(index);
}

template <typename T>
T GProperty<T>::getDomain(unsigned int index)
{
    return m_domain->getValue(index);
}

 
template <typename T>
GAry<T>* GProperty<T>::getValues()
{
    return m_values ; 
}

template <typename T>
GAry<T>* GProperty<T>::getDomain()
{
    return m_domain ; 
}






template <typename T>
GProperty<T>* GProperty<T>::load(const char* path)
{
    NPY<T>* npy = NPY<T>::load(path);
    if(!npy)
    {
        LOG(warning) << "GProperty<T>::load FAILED for path " << path ;
        return NULL ; 
    }

    //npy->Summary();

    assert(npy->getDimensions() == 2);

    unsigned int ni = npy->getShape(0);
    unsigned int nj = npy->getShape(1);
    assert(nj == 2);

    unsigned int len = ni*nj ;

    T* fdata = npy->getValues();


    // translate into local type ? no longer needed have moved to templated NPY ?
    T* data = new T[len];
    for( unsigned int i = 0 ; i < len ; i++)
    {
        data[i] = fdata[i] ;
    }

    GAry<T>* doms = new GAry<T>(ni);
    GAry<T>* vals = new GAry<T>(ni);

    for(unsigned int i=0 ; i < ni ; i++){
    for(unsigned int j=0 ; j < nj ; j++)
    {
        unsigned int index = i*nj + j ;
        T v = data[index];
        switch(j)
        {
            case 0:doms->setValue(i, v); break ; 
            case 1:vals->setValue(i, v); break ;
        }      
            //printf(" i %u j %u index %u  val %10.3f \n", i,j, index, v ); 
    }
    }

    delete data ;
    return new GProperty<T>(vals, doms) ; 
}


template <typename T>
T GProperty<T>::maxdiff(GProperty<T>* a, GProperty<T>* b)
{
    assert(a->getLength() == b->getLength());
    T mv(0);
    T md(0);
    for(unsigned int i=0 ; i < a->getLength() ; i++)
    {
        T av = a->getValues()->getValue(i) ;
        T bv = b->getValues()->getValue(i) ;
        T dv = fabs(av-bv); 

        if(dv > mv) mv = dv ;
        //printf("av %10.3f bv %10.3f dv*1e9 %10.3f mv*1e9 %10.3f \n", av, bv, dv*1e9, mv*1e9); 

        T ad = a->getDomain()->getValue(i) ;
        T bd = b->getDomain()->getValue(i) ;
        T dd = fabs(ad-bd); 
        if(dd > md) md = dd ;
        //printf("ad %10.3f bd %10.3f dd*1e9 %10.3f md*1e9 %10.3f \n", ad, bd, dd*1e9, md*1e9); 
    }

    return mv > md ? mv : md  ; 
}  




template <typename T>
GProperty<T>* GProperty<T>::from_constant(T value, T* domain, unsigned int length ) 
{
    GAry<T>* vals = GAry<T>::from_constant(length, value );
    GAry<T>* doms = new GAry<T>(length, domain);
    return new GProperty<T>( vals, doms );
}


template <typename T>
GProperty<T>* GProperty<T>::from_constant(T value, T dlow, T dhigh)
{
    unsigned int nval = 2 ; 
    T* domain = new T[nval] ; 
    domain[0] = T(dlow) ;
    domain[1] = T(dhigh) ;

    GProperty<T>* prop = from_constant(value, domain, nval );
    delete [] domain ; 
    return prop ; 
}


template <typename T>
bool GProperty<T>::hasSameDomain(GProperty<T>* a, GProperty<T>* b, T delta)
{
    if(a->getLength() != b->getLength())  return false ; 
    if(GAry<T>::maxdiff(a->getDomain(), b->getDomain()) > delta ) return false ;
    return true ; 
}



template <typename T>
GProperty<T>* GProperty<T>::make_one_minus(GProperty<T>* a)
{
    GAry<T>* doms = a->getDomain();
    GAry<T>* vals = GAry<T>::ones(a->getLength());
    vals->subtract(a->getValues());
    return new GProperty<T>( vals, doms );
} 


template <typename T>
GProperty<T>* GProperty<T>::make_addition(GProperty<T>* a, GProperty<T>* b, GProperty<T>* c, GProperty<T>* d)
{
    assert(hasSameDomain(a,b));
    if(c) assert(hasSameDomain(a,c));
    if(d) assert(hasSameDomain(a,d));

    GAry<T>* doms = a->getDomain();
    GAry<T>* vals = GAry<T>::zeros(a->getLength());
    vals->add(a->getValues());
    vals->add(b->getValues());
    if(c) vals->add(c->getValues()) ;
    if(d) vals->add(d->getValues()) ;

    return new GProperty<T>( vals, doms );
}





template <typename T>
std::string GProperty<T>::make_table(int fw, T dscale, bool dreciprocal, bool constant, std::vector< GProperty<T>* >& columns, std::vector<std::string>& titles)
{
    assert(columns.size() == titles.size());
    unsigned int ncol = columns.size();

    std::stringstream ss ; 
    if(ncol == 0) 
    {
        ss << "no columns" ; 
    }
    else
    {
        GProperty<T>* a = columns[0] ;
        for(unsigned int c=1 ; c < ncol ; c++) assert(hasSameDomain(a,columns[c]));
        GAry<T>* doms = a ? a->getDomain() : NULL ;
        assert(doms);

        ss << std::setw(fw) << "domain" ; 
        for(unsigned int c=0 ; c < ncol ; c++) ss << std::setw(fw) << titles[c] ; 
        ss << std::endl ; 

        T one(1); 
        std::vector< GAry<T>* > values ; 
        for(unsigned int c=0 ; c < ncol ; c++) values.push_back(columns[c]->getValues()) ; 

        unsigned int nr = doms->getLength(); 

        for(unsigned int r=0 ; r < nr ; r++)
        {
            if(constant && !(r == 0 || r == nr - 1)) continue ;

            T dval = doms->getValue(r) ; 
            if(dreciprocal) dval = one/dval ; 
            ss << std::setw(fw) << dval*dscale ;

            for(unsigned int c=0 ; c < ncol ; c++) ss << std::setw(fw) << ( values[c] ? values[c]->getValue(r) : -2. ) ; 
            ss << std::endl ; 
        }
    }
    return ss.str();
}


template <typename T>
std::string GProperty<T>::make_table(
       int fw, T dscale, bool dreciprocal, 
       GProperty<T>* a, const char* atitle, 
       GProperty<T>* b, const char* btitle, 
       GProperty<T>* c, const char* ctitle, 
       GProperty<T>* d, const char* dtitle, 
       GProperty<T>* e, const char* etitle, 
       GProperty<T>* f, const char* ftitle, 
       GProperty<T>* g, const char* gtitle, 
       GProperty<T>* h, const char* htitle
       )
{
    if(a && b) assert(hasSameDomain(a,b));
    if(a && c) assert(hasSameDomain(a,c));
    if(a && d) assert(hasSameDomain(a,d));
    if(a && e) assert(hasSameDomain(a,e));
    if(a && f) assert(hasSameDomain(a,f));
    if(a && g) assert(hasSameDomain(a,g));
    if(a && h) assert(hasSameDomain(a,h));

    //const char* pt = "." ; 
    const char* pt = "" ; 

    std::stringstream ss ; 
    ss << std::setw(fw) << "domain" ; 
    if(a) ss << std::setw(fw) << atitle << pt ; 
    if(b) ss << std::setw(fw) << btitle << pt ; 
    if(c) ss << std::setw(fw) << ctitle << pt ; 
    if(d) ss << std::setw(fw) << dtitle << pt ; 
    if(e) ss << std::setw(fw) << etitle << pt ; 
    if(f) ss << std::setw(fw) << ftitle << pt ; 
    if(g) ss << std::setw(fw) << gtitle << pt ; 
    if(h) ss << std::setw(fw) << htitle << pt ; 
    ss << std::endl ; 

    T one(1); 
    GAry<T>* doms = a ? a->getDomain() : NULL ;
    if(doms)
    { 
        GAry<T>* aa = a ? a->getValues() : NULL ;  
        GAry<T>* bb = b ? b->getValues() : NULL ;  
        GAry<T>* cc = c ? c->getValues() : NULL ;  
        GAry<T>* dd = d ? d->getValues() : NULL ;  
        GAry<T>* ee = e ? e->getValues() : NULL ;  
        GAry<T>* ff = f ? f->getValues() : NULL ;  
        GAry<T>* gg = g ? g->getValues() : NULL ;  
        GAry<T>* hh = h ? h->getValues() : NULL ;  

        for(unsigned int i=0 ; i < doms->getLength() ; i++)
        {
            T dval = doms->getValue(i) ; 
            if(dreciprocal) dval = one/dval ; 
            ss << std::setw(fw) << dval*dscale ; 
            if(aa) ss << std::setw(fw) << ( aa ? aa->getValue(i) : -2. ) ; 
            if(bb) ss << std::setw(fw) << ( bb ? bb->getValue(i) : -2. ) ; 
            if(cc) ss << std::setw(fw) << ( cc ? cc->getValue(i) : -2. ) ; 
            if(dd) ss << std::setw(fw) << ( dd ? dd->getValue(i) : -2. ) ; 
            if(ee) ss << std::setw(fw) << ( ee ? ee->getValue(i) : -2. ) ; 
            if(ff) ss << std::setw(fw) << ( ff ? ff->getValue(i) : -2. ) ; 
            if(gg) ss << std::setw(fw) << ( gg ? gg->getValue(i) : -2. ) ; 
            if(hh) ss << std::setw(fw) << ( hh ? hh->getValue(i) : -2. ) ; 
            ss << std::endl ; 
        }
    }
    else
    {
        ss << "EMPTY TABLE" ; 
    }

    return ss.str();
}









template <typename T>
GProperty<T>* GProperty<T>::ramp(T low, T step, T* domain, unsigned int length ) 
{
    GAry<T>* vals = GAry<T>::ramp(length, low, step );
    GAry<T>* doms = new GAry<T>(length, domain);
    return new GProperty<T>( vals, doms );
}



template <typename T>
GProperty<T>::GProperty(GProperty<T>* other) : m_length(other->getLength())
{
    m_values = new GAry<T>(other->getValues());
    m_domain = new GAry<T>(other->getDomain());
}

template <typename T>
GProperty<T>::GProperty(T* values, T* domain, unsigned int length ) : m_length(length)
{
    m_values = new GAry<T>(length, values);
    m_domain = new GAry<T>(length, domain);
}

// stealing ctor, use with newly allocated GAry<T> 
template <typename T>
GProperty<T>::GProperty( GAry<T>* vals, GAry<T>* dom )  : m_values(vals), m_domain(dom) 
{
    assert(vals->getLength() == dom->getLength());
    m_length = vals->getLength();
}

template <typename T>
GProperty<T>::~GProperty()
{
    delete m_values ;
    delete m_domain ;
} 

template <typename T>
unsigned int GProperty<T>::getLength()
{
    assert(m_values->getLength() == m_domain->getLength());
    return m_domain->getLength(); 
}


template <typename T>
void GProperty<T>::save(const char* dir, const char* reldir, const char* name)
{
    bool create = true ; 
    std::string path = BFile::preparePath(dir, reldir, name, create);
    LOG(debug) << "GProperty<T>::save to " << path ; 
    save(path.c_str());
}




template <typename T>
bool GProperty<T>::isZero()
{
    T zero(0);
    unsigned int len = getLength();
    unsigned int num(0) ;

    for(unsigned int i=0 ; i < len ; i++)
        if(m_values->getValue(i) == zero) num += 1 ; 

    return len == num ; 
}


template <typename T>
bool GProperty<T>::isConstant()
{
    unsigned int len = getLength();
    if(len == 0) return false ; 
    if(len == 1) return true ; 

    T first = m_values->getValue(0); 
    for(unsigned int i=1 ; i < len ; i++)
    {
        if(first != m_values->getValue(i)) return false ;   
    }
    return true ; 
}

template <typename T>
T GProperty<T>::getConstant()
{
    assert(getLength() > 0);
    return m_values->getValue(0); 
}


template <typename T>
void GProperty<T>::save(const char* path)
{
    std::string metadata = "{}" ; 
    std::vector<int> shape ; 
    unsigned int len = getLength();
    shape.push_back(len);
    shape.push_back(2);

    std::vector<T> data ; 
    for(unsigned int i=0 ; i < len ; i++ ){ 
    for(unsigned int j=0 ; j < 2 ; j++ )
    { 
       switch(j)
       {
           case 0:data.push_back(m_domain->getValue(i));break;
           case 1:data.push_back(m_values->getValue(i));break;
       }
    }
    }
    LOG(debug) << "GProperty::save 2d array of length " << len << " to : " << path ;  
    NPY<T> npy(shape, data, metadata);
    npy.save(path);
}



template <typename T>
void GProperty<T>::SummaryV(const char* msg, unsigned int nline )
{
   if(nline == 0) return ;
   printf("%s : \n", msg );
   for(unsigned int i=0 ; i < m_length ; i++ )
   {
      if( i < nline || i > m_length - nline )
      {
          printf("%4u", i );
          printf(DOMAIN_FMT, m_domain->getValue(i));
          printf(VALUE_FMT,  m_values->getValue(i));
          printf("\n");
      }
   }
}


template <typename T>
void GProperty<T>::Summary(const char* msg, unsigned int imod )
{
   char* pdig = digest();
   printf("%s : %s : %u \n", msg, pdig, m_length );
   free(pdig);



   for(unsigned int p=0 ; p < 2 ; p++)
   {
       switch(p)
       {
           case 0:printf("d ");break;
           case 1:printf("v ");break;
       }
       for(unsigned int i=0 ; i < m_length ; i++ )
       {
           if( i % imod == 0 )
           { 
               switch(p)
               {
                   case 0:printf(DOMAIN_FMT, m_domain->getValue(i));break;
                   case 1:printf(VALUE_FMT,  m_values->getValue(i));break;
               }
           }
       }
       printf("\n");
   }
}




template <typename T>
char* GProperty<T>::digest()
{
    size_t v_nbytes = m_values->getNbytes();
    size_t d_nbytes = m_domain->getNbytes();
    assert(v_nbytes == d_nbytes);

    SDigest dig ;
    dig.update( (char*)m_values->getValues(), v_nbytes);
    dig.update( (char*)m_domain->getValues(), d_nbytes );
    return dig.finalize();
}

template <typename T>
std::string GProperty<T>::getDigestString()
{
    return digest();
}




template <typename T>
GProperty<T>* GProperty<T>::createInterpolatedProperty(GDomain<T>* domain)
{
    GAry<T>* idom = new GAry<T>(domain->getLength(), domain->getValues());
    GAry<T>* ival = GAry<T>::np_interp( idom , m_domain, m_values );

    GProperty<T>* prop = new GProperty<T>( ival, idom );
    return prop ;
}

template <typename T>
T GProperty<T>::getInterpolatedValue(T x)
{
    // find the value "y" at "x" by first placing "x" within the domain
    // and then using linear interpolation of the above and below values
    return GAry<T>::np_interp( x , m_domain, m_values );
}

template <typename T>
GProperty<T>* GProperty<T>::createCDFTrivially()
{
   // this makes assumptions like equal bins 
    GAry<T>* x = new GAry<T>(getDomain());  // copy 
    GAry<T>* cy = getValues()->cumsum();     
    cy->scale(1./cy->getRight());            // normalise by making RHS 1.
    return new GProperty<T>( cy, x );        // stealing ctor 
}

/*
101 def construct_cdf( xy ):
102     """
103     :param xy:
104 
105     Creating cumulative density functions needed by chroma, 
106     eg for generating a wavelengths of reemitted photons.::
107 
...
146     assert len(xy.shape) == 2 and xy.shape[-1] == 2
147     x,y  = xy[:,0], xy[:,1]
148     cy = np.cumsum(y)
149     cdf_y = cy/cy[-1]   # normalize to 1 at RHS
150     return np.vstack([x,cdf_y]).T
*/



template <typename T>
GProperty<T>* GProperty<T>::createReversedReciprocalDomain()
{
    bool reciprocal = true ; 
    GAry<T>* x = getDomain()->reversed(reciprocal);   // 1/nm in reverse order 
    GAry<T>* y = getValues()->reversed();
    return new GProperty<T>( y, x );        // stealing ctor 
}

template <typename T>
GProperty<T>* GProperty<T>::createSliced(int ifr, int ito)
{
    GAry<T>* x = getDomain()->sliced(ifr, ito); 
    GAry<T>* y = getValues()->sliced(ifr, ito);
    return new GProperty<T>( y, x );        // stealing ctor
}

template <typename T>
GProperty<T>* GProperty<T>::createZeroTrimmed()
{
    GAry<T>* y = getValues();
    unsigned int ifr = y->getLeftZero();
    unsigned int ito = y->getRightZero();
   // printf("GProperty<T>::createZeroTrimmed ifr %u ito %u \n", ifr, ito);
    return createSliced(ifr, ito);
}


template <typename T>
GProperty<T>* GProperty<T>::createCDF()
{
    GAry<T>* x = getDomain()->copy();
    GAry<T>* y = getValues()->copy();

    // numerical integration of input distribution
    // * ymid, xdif, prod have one bin less as pair based 
    //
    GAry<T>* ymid = y->mid() ;    // mid bin y value
    GAry<T>* xdif = x->diff() ;   // bin widths (not constant due to reciprocation) 
    GAry<T>* prod = GAry<T>::product( ymid, xdif );  // bin-by-bin area 

    unsigned int offzero = 1 ;               // gives one extra zero bin
    GAry<T>* cy = prod->cumsum(offzero);     // cumulative summation of areas 
    cy->scale(1./cy->getRight());            // normalise CDF by making RHS 1.

    bool debug = false ; 
    if(debug)
    {
        unsigned int imod = x->getLength()/20  ; 
        T psc = 1000.f ; // presentation scale
        x->Summary("x [domain reversed reciprocal]", imod, psc);
        y->Summary("y [values reversed]", imod, psc);
        ymid->Summary("ymid: y->mid()", imod, psc);
        xdif->Summary("xdif: x->diff()", imod, psc);
        prod->Summary("prod:  ymid*xdif ", imod, psc);
        cy->Summary("cy: prod->cumsum(1) scaled  ", imod, psc);
    }

    delete y ; 
    delete ymid ;
    delete xdif ;
    delete prod ; 

    return new GProperty<T>( cy, x );        // stealing ctor 
}

//  translation of NumPy based env/geant4/geometry/collada/collada_to_chroma.py::construct_cdf_energywise
/*
154 def construct_cdf_energywise(xy):
155     """
156     Duplicates DsChromaG4Scintillation::BuildThePhysicsTable     
157     """
158     assert len(xy.shape) == 2 and xy.shape[-1] == 2
159 
160     bcdf = np.empty( xy.shape )
161 
162     rxy = xy[::-1]              # reverse order, for ascending energy 
163 
164     x = 1/rxy[:,0]              # work in inverse wavelength 1/nm
165 
166     y = rxy[:,1]
167 
168     ymid = (y[:-1]+y[1:])/2     # looses entry as needs pair
169 
170     xdif = np.diff(x)
171 
172     #bcdf[:,0] = rxy[:,0]        # back to wavelength
173     bcdf[:,0] = x                # keeping 1/wavelenth
174 
175     bcdf[0,1] = 0.
176 
177     np.cumsum(ymid*xdif, out=bcdf[1:,1])
178 
179     bcdf[1:,1] = bcdf[1:,1]/bcdf[1:,1].max()
180 
181     return bcdf
*/



template <typename T>
GProperty<T>* GProperty<T>::createInverseCDF(unsigned int n)
{
    // normally CDF values are in range 0:1 
    // with the domain being specific to the case
    //
    // InverseCDF domain is here the range 0:1 
    //
    // the objective is to avoid having to do a binary search
    // to find the bin, should be able to do a direct lookup
    //
    if(n == 0) n = getLength();
    GAry<T>* domain = GAry<T>::linspace(n, 0, 1 );
    GAry<T>* vals   = sampleCDF(domain);
    return new GProperty<T>( vals, domain ); 
}


template <typename T>
GProperty<T>* GProperty<T>::planck_spectral_radiance(GDomain<T>* nmdom, T blackbody_temp_kelvin)
{
    GAry<T>* nm = GAry<T>::from_domain(nmdom );

    GAry<T>* vals   = GAry<T>::planck_spectral_radiance( nm, blackbody_temp_kelvin );

    return new GProperty<T>( vals, nm ); 
}




template <typename T>
GAry<T>* GProperty<T>::lookupCDF(unsigned int n)
{
    GAry<T>* ua = GAry<T>::urandom(n); 
    return lookupCDF(ua);
}

template <typename T>
GAry<T>* GProperty<T>::lookupCDF_ValueLookup(unsigned int n)
{
    GAry<T>* ua = GAry<T>::urandom(n); 
    return lookupCDF_ValueLookup(ua);
}




template <typename T>
GAry<T>* GProperty<T>::lookupCDF(GAry<T>* ua)   // start from domain, not values
{
    unsigned int len = ua->getLength();
    T* u = ua->getValues();
    GAry<T>* sample = new GAry<T>(len); 
    for(unsigned int i=0 ; i < len ; i++)
    {
        T x = u[i] ;                     // domain value 
        T y = getInterpolatedValue(x);
        sample->setValue(i,  y);
    }
    return sample ; 
}


template <typename T>
GAry<T>* GProperty<T>::lookupCDF_ValueLookup(GAry<T>* ua)
{
    unsigned int len = ua->getLength();
    T* u = ua->getValues();
    GAry<T>* sample = new GAry<T>(len); 
    for(unsigned int i=0 ; i < len ; i++)
    {
        sample->setValue(i,  m_values->getValueLookup(u[i]));
    }
    return sample ; 
}




template <typename T>
GAry<T>* GProperty<T>::sampleCDF(unsigned int n)
{
    GAry<T>* ua = GAry<T>::urandom(n); 
    return sampleCDF(ua);
}
 
template <typename T>
GAry<T>* GProperty<T>::sampleCDF(GAry<T>* ua)  // start from values, not domain
{
    unsigned int len = ua->getLength();
    T* u = ua->getValues();
    GAry<T>* sample = new GAry<T>(len); 
    for(unsigned int i=0 ; i < len ; i++)
    {
         T x = u[i] ;                                    // domain value (normally in range 0 to 1)
         T f = m_values->fractional_binary_search(x);    // fractional bin index corresponding to x
         T d = m_domain->getValueFractional(f);
         sample->setValue(i,  d );

         // hmm this is like interpolation but with a swap: values <-> domain
    }
    return sample ; 
}

template <typename T>
GAry<T>* GProperty<T>::sampleCDFDev(unsigned int n)
{
    GAry<T>* ua = GAry<T>::urandom(n); 
    GAry<T>* sample = new GAry<T>(n); 

    T* values = m_values->getValues();

    for(unsigned int i=0 ; i < ua->getLength() ; i++)
    {
         T u = ua->getValue(i); 
         T f = m_values->fractional_binary_search(u);

         {
             unsigned int idx2 = m_values->sample_cdf(u);
             unsigned int idx = m_values->binary_search(u);
             assert(idx == idx2);
             unsigned int fi(f);
             //assert(fi == idx );


             T ulo    = values[idx];
             T uhi    = values[idx+1];
             T udelta = values[idx+1] - values[idx] ;
             T uoff   = u - values[idx] ;
             T ufrac  = (u - values[idx])/(values[idx+1]-values[idx]);
             T ff = T(idx) + ufrac ; 

             if(fi != idx)  // a few with ufrac almost 1 dont match
             printf("i %u  idx %u      ulo %15.6f u %15.6f uhi %15.6f udelta %15.6f uoff %15.6f ufrac %15.6f f %15.6f ff %15.6f  \n", i, idx, ulo, u, uhi, udelta, uoff, ufrac, f, ff  );

         }


         // NB sampling depends on the values only up to here, 
         //    domain only comes in at end to convert the fractional value index 
         //    (sort of bin number) into a domain value

         /*
         unsigned int idx(f);
         T dlo = m_domain->getValue(idx);
         T dhi = m_domain->getValue(idx+1);
         T frac(f - T(idx));
         T dva = dlo + (dhi - dlo)*frac ;
         */

         T dva = m_domain->getValueFractional( f );


         //  TODO: check fractional bin  
         sample->setValue(i,  dva  );
    }
    delete ua ; 
    return sample ; 
}




/*
* :google:`move templated class implementation out of header`
* http://www.drdobbs.com/moving-templates-out-of-header-files/184403420

A compiler warning "declaration does not declare anything" was avoided
by putting the explicit template instantiation at the tail rather than the 
head of the implementation.
*/

template class GProperty<float>;
template class GProperty<double>;

