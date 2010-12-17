r"""

\begin{document}
\begin{itemize}
\item red
\item green
\item blue
\end{itemize}

Some content
\begin{tabular}{ccc}
 one & two & three \\
 red & green & blue \\
\end{tabular}


\end{document}


"""
from cStringIO import StringIO
import sys
from convert import _convert_file
_convert_file( StringIO(__doc__) , sys.stdout )





