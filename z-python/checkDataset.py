import os

testDir = "datasets/VOC2007/images/test/"
trainDir = "datasets/VOC2007/images/train/"
valDir = "datasets/VOC2007/images/val/"

testNames = os.listdir(testDir)
trainNames  = os.listdir(trainDir)
valNames = os.listdir(valDir)

testInTrain = 0
testInVal = 0
for  i, imgName in enumerate(testNames):
    if imgName in trainNames:
        testInTrain+=1
    if imgName in valNames:
        testInVal +=1

print("test image num: ", len(testNames))
print("test In Train: ", testInTrain)
print("test In Val", testInVal)


valInTrain = 0
valInTest = 0
for  i, imgName in enumerate(valNames):
    if imgName in trainNames:
        valInTrain+=1
    if imgName in testNames:
        valInTest +=1

print("val  image num: ", len(valNames))
print("val In Train: ", valInTrain)
print("val In Test: ", valInTest)

trainInTest = 0
trainInVal = 0
for  i, imgName in enumerate(trainNames):
    if imgName in testNames:
        trainInTest+=1
    if imgName in valNames:
        trainInVal +=1

print("train  image num: ", len(trainNames))
print("train In Test: ", trainInTest)
print("train In Val: ", trainInVal)