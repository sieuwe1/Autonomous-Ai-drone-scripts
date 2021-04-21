import matplotlib.pyplot as plt

file1 = open('new_drone_1_Depth.txt', 'r')
Lines = file1.readlines()

data = []

line_count = 0

for line in Lines:
    if line_count > 0:
        section_count = 0
        sections = line.split(',') 
        print(sections)
        
        if line_count == 1:
            for i in range(len(sections)):
                data.append([])

        for section in sections:
            if '\n' in section:
                section = section[0:-1]
            point = float(section)
            print(point)
            if point != 0:
                data[section_count].append(point)
            
            section_count+=1

    line_count+=1
    
print(len(data[4]))

plt.plot(data[3])
plt.plot(data[4])
plt.ylabel('some numbers')
plt.show()