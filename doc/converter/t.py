
from converter import convert_file
from tempfile import NamedTemporaryFile as T
tex1 = """\begin{document}
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

i,o = T(),T()
i.write(tex1)

convert_file( i.name , o.name )

print o.read()

