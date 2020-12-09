# remove duplicated images
import shutil

readDir = "venice_m1.txt"

writeDir = "m1_dd.txt"

lines_seen = set()

outfile=open(writeDir,"w")

f = open(readDir,"r")

for line in f:
	if line not in lines_seen:
		outfile.write(line)
		lines_seen.add(line)
outfile.close()
print("success")

## result min lat- max lat: 45.404235-45.46308
##        min lon- max lon: 12.258982-12.377772