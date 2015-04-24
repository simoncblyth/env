from docx import Document

document = Document()

styles = document.styles

print styles
for s in styles:print s


document.save('test.docx')
