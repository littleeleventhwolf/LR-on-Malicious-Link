import csv
c = open("data.csv", "r")
read = csv.reader(c)

p = open("datacp.csv", "w")
write = csv.writer(p)

for line in read:
    if(line[1] == 'bad'):
            line[1] ='0'
    else:
            line[1] = '1'
    write.writerow([line[0], line[1]])

