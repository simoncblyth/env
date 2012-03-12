"""
file:///opt/local/share/doc/py25-cython/About.html

Build extension with::

   python setup.py build    ## cython compiles this primes.pyx into primes.c and compiles that into extenstion module

Test inplace::

	g4pb:cy blyth$ PYTHONPATH=build/lib.macosx-10.5-ppc-2.5 python -c "import primes ; print primes.primes(10)"
	[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


"""
def primes(int kmax):
    cdef int n, k, i
    cdef int p[1000]
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] <> 0:
            i = i + 1
        if i == k:
           p[k] = n
           k = k + 1
           result.append(n)
        n = n + 1
    return result
