import os

curPath=os.getcwd()
tempPath='name'
targetPath=curPath+os.path.sep+tempPath
print(os.path.sep)
if not os.path.exists(targetPath):
    os.makedirs(targetPath)
else:
    print('路径已经存在！')
fileName='abc.txt'
filePath=targetPath+os.path.sep+fileName
with open(filePath,'w') as f:
    f.write('Hello world!')
    print('写入成功！')