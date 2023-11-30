import os

file1 = open('trainSamples.txt', 'w')
file2 = open('trainLabels.txt', 'w')
for j in range(10):
    path = 'train/'+str(j)
    list_ = os.listdir(path)
    
    for i in range(len(list_)):
        if(list_[i].find('jpg')>0):
           path_file = path + "/" +list_[i]+"\n"
           file1.write('mnist/'+path_file)
           file2.write(str(j)+'\n')