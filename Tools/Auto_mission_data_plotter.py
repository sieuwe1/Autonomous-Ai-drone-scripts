#plot data from auto mission
from matplotlib import lines
import matplotlib.pyplot as plt

file1 = open('/home/drone/Desktop/Autonomous-Ai-drone-scripts/predictions_log.txt', 'r')
Lines = file1.readlines()

sections = Lines[0].split(',') 

filtered = [x for x in sections if "dtype" not in x]

data_raw = []
for i in range(4,len(filtered)-3,4):
    data_raw.append(str(filtered[i-4]) + "," + str(filtered[i-3]) + "," + str(filtered[i-2]) + "," + str(filtered[i-1]))


print()
print(data_raw[0])
print(data_raw[1])
print(data_raw[2])
print(data_raw[3])
print(data_raw[4])
print(data_raw[5])
print(data_raw[6])
print(data_raw[7])


data = []
#for i in range(len(data_raw)):
    #print(data_raw)
    #print(data_raw[i][0].split('[')[3].split(']')[0])
    #print(data_raw[i][1].split('[')[2].split(']')[0])
    #print(data_raw[i][2].split('[')[2].split(']')[0])
    #print(data_raw[i][3].split('[')[2].split(']')[0])
    

    #data.append(data_raw[i][0].split('[')[3].split(']')[0], data_raw[i][1].split('[')[3].split(']')[0], 
    #data_raw[i][2].split('[')[3].split(']')[0], data_raw[i][3].split('[')[3].split(']')[0])
       
#print(data)
#for line in Lines:
    #print(line)
    #sections = line.split(',') 
    #print(sections)


#plt.plot(data)
#plt.ylabel('some numbers')
#plt.show()
