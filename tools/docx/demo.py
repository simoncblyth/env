#!/usr/bin/env python
"""
https://python-docx.readthedocs.io/en/latest/



"""

from docx import Document
from docx.shared import Inches
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.style import WD_STYLE


document = Document()

document.add_heading('Document Title', 0)

p = document.add_paragraph('A plain paragraph having some ')
p.add_run('bold').bold = True
p.add_run(' and some ')
p.add_run('italic.').italic = True

document.add_heading('Heading, level 1', level=1)
document.add_paragraph('Intense quote', style='IntenseQuote')

document.add_paragraph(
    'first item in unordered list', style='ListBullet'
)
document.add_paragraph(
    'first item in ordered list', style='ListNumber'
)

#document.add_picture('monty-truth.png', width=Inches(1.25))

table = document.add_table(rows=1, cols=3)
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Qty'
hdr_cells[1].text = 'Id'
hdr_cells[2].text = 'Desc'

#for item in recordset:
#    row_cells = table.add_row().cells
#    row_cells[0].text = str(item.qty)
#    row_cells[1].text = str(item.id)
#    row_cells[2].text = item.desc




## http://stackoverflow.com/questions/27884703/set-paragraph-font-in-python-docx



## hmm the below changes everything not just pp
#style = document.styles['Normal']
#font = style.font
#font.name = 'Arial'
#font.size = Pt(20)
#pp.style = document.styles['Normal']


txt = r"""

.       1000000   1000000   373.13/356 =  1.05  (pval:0.256 prob:0.744)  

0000     669843    670001     0.02     TO BT BT BT BT SA
0001      83950     84149     0.24     TO AB
0002      45490     44770     5.74     TO SC BT BT BT BT SA
0003      28955     28718     0.97     TO BT BT BT BT AB
0004      23187     23170     0.01     TO BT BT AB
0005      20238     20140     0.24     TO RE BT BT BT BT SA
0006      10214     10357     0.99     TO BT BT SC BT BT SA
0007      10176     10318     0.98     TO BT BT BT BT SC SA
0008       7540      7710     1.90     TO BT BT BT BT DR SA
0009       5976      5934     0.15     TO RE RE BT BT BT BT SA
0010       5779      5766     0.01     TO RE AB

"""



mls = document.styles.add_style('MyListingStyle', WD_STYLE_TYPE.PARAGRAPH)
font = mls.font
pfmt = mls.paragraph_format

font.name = 'Courier New'
font.size = Pt(8)

pfmt.left_indent = Inches(-3.0)
pfmt.right_indent = Inches(-3.0)


pp = document.add_paragraph(txt, style=mls)

#pp.add_run("this is in CommentsStyle bolded", style = 'CommentsStyle').bold = True
#pp.add_run("this is in CommentsStyle not bolded", style = 'CommentsStyle')


lnk = document.styles[WD_STYLE.HYPERLINK]

pp.add_run("http://on-demand.gputechconf.com/gtc/2016/video/s6320-simon-blyth-opticks-nvidia-optix.mp4", style=lnk )

document.add_page_break()

document.save('/tmp/demo.docx')
