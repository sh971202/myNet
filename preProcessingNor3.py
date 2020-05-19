import sys
import argparse
import string
import time
import cv2
import math
import numpy as np


#from tf.transformation import quaternion_matrix
from random import sample 
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

np.set_printoptions(suppress=True)

decimals = 4
errorThreshold = 10

correct = 0
wrong = 0
miss = 0
correctR = 0
wrongR = 0
missR = 0

class GroundTruthCorre():
    def __init__(self, imageName, focalLength, quaternion, rotationMatrix, cameraCenter, distortion):
        
        self.imageName = imageName
        self.corres = []

        self.focalLength = focalLength
        #self.rotationMatrix = np.transpose(rotation.as_matrix())
        #self.rotationMatrix = inv(rotation.as_matrix())
        #self.rotationMatrix = rotation.as_matrix()
        #self.rotationVector = rotation.as_rotvec()

        self.quaternion = quaternion
        self.rotationMatrix = rotationMatrix

        self.cameraCenter = cameraCenter
        self.distortion = distortion

        self.transformVector = np.dot(-(self.rotationMatrix), self.cameraCenter)
        #self.transformVector = np.dot(-(inv(self.rotationMatrix)), self.cameraCenter)

        self.srcPoints = []
        self.desPoints = []

        self.distortionCoeffs = np.array([])
        
        self.intrinsicMatrix = np.array([[focalLength, 0.0, 0.0],
                                        [0.0, focalLength, 0.0],
                                        [0.0, 0.0, 1.0]])
    
        RT = np.concatenate((self.rotationMatrix, np.expand_dims(self.transformVector, axis = 0).T), axis = 1)
        self.projectMatrix = np.dot(self.intrinsicMatrix, RT)

        '''
        if imageName[0:2] == '38':
            print ("imageName: \n", self.imageName, "\n")
            print ("focalLength: \n", self.focalLength, "\n")
            print ('Q2M: ', rotationMatrix, '\n')
            print ("RT: \n", RT, "\n")
            print ("projectMatrix: \n", self.projectMatrix, "\n")
            print ("cameraCenter: \n", self.cameraCenter, "\n")
        '''


    def addCorres(self, cord2D, cord3D):  # corres = x, y, x, y, z
        
        self.corres.append([ cord2D, cord3D ])
        self.srcPoints.append(cord3D)
        self.desPoints.append(cord2D)

    def checkCord(self, test, groundtruth):

        same = True
        '''
        for testCord, groundtruthCord in zip(test, groundtruth):
            
            if (abs(testCord) > abs(groundtruthCord) + 2) or (abs(testCord) < abs(groundtruthCord) - 2):
                same = False
        '''
        if (test[0] > 0) == (groundtruth[0] > 0):
            if (test[0] > groundtruth[0] + 2) or (test[0] < groundtruth[0] - 2):
                same = False
            if (test[1]) > -groundtruth[1] + 2 or (test[1] < -groundtruth[1] - 2):
                same = False
        else:
            if (test[0] > -groundtruth[0] + 2) or (test[0] < -groundtruth[0] - 2):
                same = False
            if (test[1]) > groundtruth[1] + 2 or (test[1] < groundtruth[1] - 2):
                same = False
        return same

    def getLabelBySearch(self, cord2D, cord3D):

        #2Dcord = testCorre[0:1]
        #3Dcord = testCorre[2:]
        global correct, wrong, miss
        same3Dcord = False

        for corre in self.corres:
            #print (cord2D, corre[0], cord3D, corre[1])
            #time.sleep(.05)
            if np.any(cord3D != corre[1]):
                #print ('Different 3D cord')
                continue
            elif self.checkCord(cord2D, corre[0]):
                # Condition of true or false here
                #print ("Correct Correspondence")
                correct += 1
                #time.sleep(1)
                return True
            else:
                # Different 2D cord
                same3Dcord = True
        
        if not same3Dcord:
            #print ("Didn't find same 3D cord")
            miss += 1
        else:
            wrong += 1
            #print ('False Correspondence')
        return False    

    def getLabelByReprojection(self, cord2D, cord3D):

        global correctR, wrongR, missR, errorThreshold

        xChange = False
        yChange = False
    
        #cord3D = np.array([cord3D[2], cord3D[1], cord3D[0]])

        #print (cord3D, self.rotationVector, self.cameraCenter, self.intrinsicMatrix)
        #estimatePoint = (0, 0)
        
        objPoint = np.expand_dims(cord3D, axis = 0)
        
        #estimatePoint, jaco = cv2.projectPoints(objPoint, self.rotationVector, self.transformVector, self.intrinsicMatrix, None)
        #estimatePoint, jaco = cv2.projectPoints(objPoint, self.rotationVector, self.cameraCenter, self.intrinsicMatrix, np.array([0.045539, -0.057822, 0.001451, -0.000487, 0.006539, 0.438100, -0.135970, 0.011170]))
        #estimatePoint = np.squeeze(np.array(estimatePoint).astype(np.float32))
        #estimatePoint = np.multiply(estimatePoint, np.array([-1, -1])) + np.array([720, 540])

        #print (cord3D, self.rotationMatrix, self.transformVector)
        myPoint = np.dot(self.rotationMatrix, cord3D.T) + self.transformVector
        myPoint = np.dot(self.intrinsicMatrix, myPoint)
        myPoint = np.array([myPoint[0] / myPoint[2], (myPoint[1] / myPoint[2])])

        #myPoint = np.multiply(myPoint, np.array([-1, -1])) + np.array([0, 0])
        #myPoint = np.array([myPoint[0], myPoint[1]])

        # undistortion
        r2 = self.distortion * (cord2D ** 2).sum()
        r2 = r2 / (self.focalLength ** 2)
        undistorted2D = np.multiply(np.array([1 + r2, 1 + r2]), cord2D)

        if (myPoint[0] > 0 ) != (undistorted2D[0] > 0):
            myPoint[0] = -myPoint[0]
            xChange = True
        if (myPoint[1] > 0 ) != (undistorted2D[1] > 0):
            myPoint[1] = -myPoint[1]
            yChange = True

        reprojectionError = math.sqrt(((undistorted2D - myPoint)**2).sum())
        #reprojectionError = abs(undistorted2D[1] - myPoint[1])

        #print (undistorted2D, myPoint, reprojectionError, xChange, yChange)
        #print (cord2D, undistorted2D, estimatePoint, myPoint, reprojectionError)
        #print ()

        if reprojectionError <= errorThreshold:
            #print ('Correct Correspondence')
            #print (undistorted2D, myPoint, reprojectionError, xChange, yChange, self.imageName)
            correctR += 1
            return True
        else:
            #print ('False Correspondence')
            wrongR += 1
            return False

    def cameraCalibration(self):

        self.srcPoints = np.expand_dims(np.array(self.srcPoints).astype(np.float32), axis = 0)
        self.desPoints = np.expand_dims(np.array(self.desPoints).astype(np.float32), axis = 0)

        print (self.srcPoints)

        #retval, self.intrinsicMatrix, self.distortionCoeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.srcPoints, self.desPoints, (1440, 1080), self.intrinsicMatrix, None)
        out = cv2.calibrateCamera(self.srcPoints, self.desPoints, (1440, 1080), self.intrinsicMatrix, self.distortionCoeffs, self.rvecs, self.tvecs, CV_CALIB_USE_INTRINSIC_GUESS)


    
def quaternionToMatrix(quaternion):

    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]

    Matrix = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*z*x+2*y*w],
                        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

    return Matrix

def main():

    global decimals, correct, wrong, miss

    minx, maxx = 0, 0
    miny, maxy = 0, 0
    minz, maxz = 0, 0

    firstData = True

    args = argParse()

    f = open(args.dataPath + '/data.txt', 'r')
    gt = open(args.dataPath + '/groundtruth.nvm', 'r')
    correbuffile = open(args.dataPath + '/correbuf.txt', 'a')
    trainfile = open(args.dataPath + '/train_nor.txt', 'a')
    testfile = open(args.dataPath + '/test_nor.txt', 'a')

    groundTruthList = [] 

    # Read gt txt
    print ('Reading GroundTruth......')
    for lineCount, line in enumerate(gt.readlines()):
        # lineCount start from 0
        #print (lineCount)
        if lineCount <= 1:
            continue

        elif lineCount == 2:
            imageNum = int(line)
            imageNameDic = {}
            imageIDDic = {}

        elif lineCount <= 2 + imageNum:
            line = line.split()

            # read imageName, camera pose
            imagePathName = line[0].split('/')
            imageName = imagePathName[-1][:-4]
            #print (imageName)
            
            line = np.array(line)[1:].astype(np.float32)
            focalLength = line[0]
            quaternion = line[1:5]
            cameraCenter = line[5:8]
            distortion = line[8]

            #rotation = R.from_quat(quaternion)

            rotationMatrix = quaternionToMatrix(quaternion)


            #print (imageName)

            groundTruthList.append(GroundTruthCorre(imageName,
                focalLength, quaternion, rotationMatrix, cameraCenter, distortion))
            imageNameDic.update({lineCount-3 : imageName}) # imageID : imageName
            imageIDDic.update({imageName : lineCount-3})   # imageName : imageID

        elif lineCount == 4 + imageNum:
            pointNum = int(line)

        elif lineCount > 4 + imageNum:

            # end of bundle
            if len(line) < 2:
                break

            line = np.array(line.split())
            line = line.astype(np.float32)
            #line = np.around(line.astype(np.float32), decimals = decimals)
            #read point, corre
            cord3D = line[:3]
            rgb = line[3:5]
            num2D = int(line[6])

            if firstData:
                minx, maxx = cord3D[0], cord3D[0]
                miny, maxy = cord3D[1], cord3D[1]
                minz, maxz = cord3D[2], cord3D[2]
                firstData = False

            minx = cord3D[0] if cord3D[0] < minx else minx
            maxx = cord3D[0] if cord3D[0] > maxx else maxx
            miny = cord3D[1] if cord3D[1] < miny else miny
            maxy = cord3D[1] if cord3D[1] > maxy else maxy
            minz = cord3D[2] if cord3D[2] < minz else minz
            maxz = cord3D[2] if cord3D[2] > maxz else maxz



            for i in range(num2D):
                imageID = int(line[7 + i*4]) # each i has imageID, kpID, x, y
                kpID = int(line[8 + i*4])
                cord2D = line[9 + i*4: 11 + i*4]
                
                groundTruthList[imageID].addCorres(cord2D, cord3D)
            
                # for debug
                #if imageNameDic[imageID] == '100':
                    #print (cord2D)
    #Camera Calibrate

    #for groundtruth in groundTruthList:
    #    groundtruth.cameraCalibration()

    #Read data txt
    print ('Reading Data......')
    trainCount = 0
    testCount = 0
    trainCorrect = 0
    testCorrect = 0
    trainWrong = 0
    testWrong = 0
    imgCount = 0
    imageName = ''
    for lineCount, line in enumerate(f.readlines()):
        correCount = 0
        line = line.split()
        
        if len(line) < 2:
            continue
        
        elif line[0][:6] == 'vector':
            # RANSAC inliers
            ransacInliers = []
            correCount = 0
            for inlier in line[2:-1]:
                ransacInliers.append(int(inlier.replace(',', '')))
            continue

        imagePathName = line[0].split('/')
        imgCount += 1 if imageName != imagePathName[-1][:-4] else 0
        imageName = imagePathName[-1][:-4]
        #print (imageName)
        #imageName = str(int(imageName) + 288)

        line = np.array(line[1:]).astype(np.float32)

        cord2D = line[:2]
        cord3D = line[2:]

        #labelBySearch = groundTruthList[imageIDDic[imageName]].getLabelBySearch(cord2D, 
        #       cord3D)
        labelByReprojection = groundTruthList[imageIDDic[imageName]].getLabelByReprojection(cord2D, 
                cord3D)
        focalLength = groundTruthList[imageIDDic[imageName]].focalLength
        quaternion = np.array2string(\
                groundTruthList[imageIDDic[imageName]].quaternion, separator =\
                ' ')
        transformVector = np.array2string(\
                groundTruthList[imageIDDic[imageName]].transformVector,\
                separator = ' ')

        ransacLabel = True if correCount in ransacInliers else False
        correCount += 1

        #if label is not 'Error':

        cord3D[0] = (cord3D[0] - minx) / (maxx - minx)
        cord3D[1] = (cord3D[1] - miny) / (maxy - miny)
        cord3D[2] = (cord3D[2] - minz) / (maxz - minz)
        
        if imgCount % 5 == 4:
            # test file
            if labelByReprojection == True:
                if testCount % 4 == 3:
                    print (imageName+'_3_sh',\
                    cord2D[0], cord2D[1], cord3D[0], cord3D[1], cord3D[2],\
                    labelByReprojection, ransacLabel, focalLength, quaternion[1:-1],\
                    transformVector[1:-1], file = testfile)
                    testCorrect += 1
                testCount += 1
            else:
                print (imageName+'_3_sh',\
                cord2D[0], cord2D[1], cord3D[0], cord3D[1], cord3D[2],\
                labelByReprojection, ransacLabel, focalLength, quaternion[1:-1],\
                transformVector[1:-1], file = testfile)
                testWrong += 1
        else:
            # train file
            #print (imageName,\
            #cord2D[0], cord2D[1], cord3D[0], cord3D[1], cord3D[2],\
            #labelByReprojection, ransacLabel, focalLength, quaternion[1:-1],\
            #transformVector[1:-1], file = correbuffile)
            
            if labelByReprojection == True:
                if trainCount % 4 == 3:
                    print (imageName+'_3_sh',\
                    cord2D[0], cord2D[1], cord3D[0], cord3D[1], cord3D[2],\
                    labelByReprojection, ransacLabel, focalLength, quaternion[1:-1],\
                    transformVector[1:-1], file = trainfile)
                    trainCorrect += 1
                trainCount += 1
            else:
                print (imageName+'_3_sh',\
                cord2D[0], cord2D[1], cord3D[0], cord3D[1], cord3D[2],\
                labelByReprojection, ransacLabel, focalLength, quaternion[1:-1],\
                transformVector[1:-1], file = trainfile)
                trainWrong += 1
            
    '''
    correbuffile.close()
    correbuffile = open(args.dataPath + '/correbuf.txt', 'r')
    print ('Sampling......')
    trueCorre = [] 
    for lineCount, line in enumerate(correbuffile.readlines()):

        corre = line.split()
        #print (corre[6])
        label = 1 if (corre[6] == 'True') else 0

        if label == 0:
            print (line.rstrip(), file = trainfile)
            trainWrong += 1
        else:
            trueCorre.append(line)

    print (len(trueCorre))
    sampleNumber = int(len(trueCorre) / 4)
    for corre in sample(trueCorre, sampleNumber):

        print (corre.rstrip(), file = trainfile)
        trainCorrect += 1
    '''
        



    trainfile.close()

    print ('trainCorrect : %d, trainWrong : %d' % (trainCorrect, trainWrong))
    print ('testCorrect : %d, testWrong : %d' % (testCorrect, testWrong))
    print ('By Search Correct: %d, Wrong: %d, Miss: %d' % (correct, wrong, miss))
    print ('By Reprojection Correct: %d, Wrong: %d, Miss: %d' % (correctR, wrongR, missR))

    return


def argParse():

    parser = argparse.ArgumentParser()

    parser.add_argument('dataPath',
                        help = 'Correspondence & groundtruth dir path')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
