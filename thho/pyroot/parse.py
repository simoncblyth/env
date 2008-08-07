

for l in file("test.txt").readlines():
    if l[0:2] == "FSR":
        print l
    else:
        aa=l.split("\t")
        print aa




