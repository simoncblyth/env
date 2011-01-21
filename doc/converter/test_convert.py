#!/usr/bin/env python
r"""

\section{Introduction}

The Electronics Simulation is in the ElecSim package.
It takes an SimHeader as input and produces an ElecHeader, which
will be read in by  the Trigger Simulation package.  The position
where ElecSim fit in the full simulation chain is given in
figure~\ref{fig-electronics-simChain}. The data model
used in ElecSim is summarized in the UML form
 in figure~\ref{fig-electronics-ElecSimUML}. 

\begin{figure}[ht]
\begin{center}
\includegraphics[width=0.98\textwidth]{../fig/electronics_simChain}
\caption{\label{fig-electronics-simChain}}
\end{center}
\end{figure} 

\begin{figure}[ht]
\begin{center}
\includegraphics[width=1.00\textwidth]{../fig/electronics_ElecSimUML}
\caption{\label{fig-electronics-ElecSimUML} UML for data model in ElecSim.}
\end{center}
\end{figure}


"""


verbatim_etc = r"""

\begin{document}

To use this you will need to hookup to a python with \code{converter} pkg. 

The hookup can be done either, by getting into the virtualenv :
\begin{verbatim}
. ~/rst/bin/activate
./test_convert.py
\end{verbatim}

Or by using the appropriate python :
\begin{verbatim}
~/rst/bin/python test_convert.py 
\end{verbatim}


Another verbatim check ..
\begin{verbatim}
a
 b 
  c
   d
\end{verbatim}


This part will explain how exactly to write your own simulation script with Fifteen package.
The example is from dybgaudi/Tutorial/Sim15/aileron/FullChainSimple.py which implements all the
basic elements.

\begin{lstlisting}[language=python]
#!/usr/bin/env python
'''
Configure the full chain of simulation from kinematics to readouts and
with multiple kinematics types mixed together.

usage:
    nuwa.py -n50 -o fifteen.root -m "FullChainSimple -T SingleLoader" > log

    -T: Optional stages are: Kinematic, Detector, Electronic, TrigRead or SingleLoader.

    More options are available like -w: wall clock starting time
                                    -F: time format
                                    -s: seed for IBD generator

    //////
    Aside:
    This is a copy of MDC09b.runIBD15.FullChain, however with less options,
    less generators configured and less truth info saved.
    //////
'''
\end{lstlisting}

This is the first part of this script. In the first line it declares the running
environment. What follows, quoted by ''', are a brief introduction of this script and usage
of this script. It tells that this script will configure a full chain of simulation.
It also includes a command line which can be used right away to start.
Before looking into the script it also explains what arguments can be set and
what are their options. These arguments will explained later.





\begin{itemize}
\item red
\item green
\item blue
\end{itemize}


hello & worlf
hello ~ worlf

Some content
\begin{tabular}{ccc}
 one & two & three \\
 red & green & blue \\
 cyan  & magenta & yellow \\
 itsy & \emph{bitsy} and ditsy & yellow \\
 polka & dot & bikini  \\
\end{tabular}


\end{document}


"""
from cStringIO import StringIO
import sys
from convert import _convert_file
_convert_file( StringIO(__doc__) , sys.stdout )





