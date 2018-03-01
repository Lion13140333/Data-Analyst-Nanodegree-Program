
#将nodes.csv转成put.csv,将空行处理掉
import csv
input = open(r'C:\\Users\\Administrator\\nodes.csv', 'rb')
output = open(r'C:\\Users\\Administrator\\put.csv', 'wb')
writer = csv.writer(output)
for row in csv.reader(input):
    if any(row):
        writer.writerow(row)
input.close()
output.close()
