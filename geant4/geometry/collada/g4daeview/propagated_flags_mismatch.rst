
Propagated Flags Mismatch
============================

Comparing propagation results is showing a rare difference in the
step from material. Reproduce with::

   delta:~ blyth$ cd /usr/local/env/tmp/1
   delta:1 blyth$ daephotonscompare.sh --loglevel debug

The npz files being compared are pulled off the GPU into numpy arrays.


Nine out of ~5000::

    In [3]: np.all( a.propagated['flags'] == b.propagated['flags'] )
    Out[3]: False

    In [4]: np.where( a.propagated['flags'] != b.propagated['flags'] )
    Out[4]: 
    (array([32990, 32998, 32999, 
            33180, 33188, 33189, 
            33460, 33468, 33469]),
     array([3, 3, 3, 3, 3, 3, 3, 3, 3]))

    In [5]: indice = np.where( a.propagated['flags'] != b.propagated['flags'] )

    In [6]: a.propagated['flags'][indice]
    Out[6]: array([3, 3, 3, 8, 8, 8, 8, 8, 8], dtype=uint32)

    In [7]: b.propagated['flags'][indice]
    Out[7]: array([4, 4, 4, 4, 4, 4, 5, 5, 5], dtype=uint32)


Discrepants all have the same pattern with the start material code differing::

    In [8]: a.propagated['flags'][32990:32990+10]
    Out[8]: 
    array([[         1, 1103839338, 1103839338,          3],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         1, 1103839338, 1103839338,          3],
           [         1, 1103839338, 1103839338,          3]], dtype=uint32)

    In [9]: b.propagated['flags'][32990:32990+10]
    Out[9]: 
    array([[         1, 1103839338, 1103839338,          4],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         0,          0,          0,          0],
           [         1, 1103839338, 1103839338,          4],
           [         1, 1103839338, 1103839338,          4]], dtype=uint32)


Have seen issues of varying material codes on invokation, but this
appears not to be that, as the numbers of discrepancies is low and
not covering all instances of the code::


    In [18]: len(np.where( a.propagated['flags'][:,3] == 3 )[0])
    Out[18]: 6319

    In [19]: len(np.where( b.propagated['flags'][:,3] == 3 )[0])
    Out[19]: 6316

    In [21]: len(np.where( a.propagated['flags'][:,3] == 4 )[0])
    Out[21]: 2172

    In [20]: len(np.where( b.propagated['flags'][:,3] == 4 )[0])
    Out[20]: 2178

    In [22]: len(np.where( a.propagated['flags'][:,3] == 5 )[0])
    Out[22]: 11204

    In [23]: len(np.where( b.propagated['flags'][:,3] == 5 )[0])
    Out[23]: 11207




