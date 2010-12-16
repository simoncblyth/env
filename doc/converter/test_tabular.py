
from converter import convert_file
from tempfile import NamedTemporaryFile as T

base = os.environ.get('DYB')
texdir = os.path.join( texdir, 'database' )
outdir = '/tmp/out'



tex1 = """

\begin{document}
\begin{itemize}
\item red
\item green
\item blue
\end{itemize}
\end{document}

"""

tex2 = """
Some content
\begin{tabular}{ccc}
 one & two & three \\
 red & green & blue \\
\end{tabular}

"""



i = T()
i.write(tab)
o = T()
convert_file( i.name , o.name )

print o.read()




