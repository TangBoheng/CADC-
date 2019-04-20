import os
train_file = open('/home/t/Desktop/bing/720/720train.txt','w')
test_file = open('/home/t/Desktop/bing/720/720test.txt','w')
for num in range(0,280):
    #print("/home/t/Desktop/bing/716train/"+str(num).zfill(6)+".jpg"+'\n')
    train_file.write("/home/t/Desktop/bing/720/720train/" + str(num).zfill(6) + ".jpg" + '\n')

for num in range(281, 317):
    # print("/home/t/Desktop/bing/716train/"+str(num).zfill(6)+".jpg"+'\n')
    test_file.write("/home/t/Desktop/bing/720/720train/" + str(num).zfill(6) + ".jpg" + '\n')
