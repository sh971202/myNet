import sys
sys.path.append('/lib/python3.6/site-packages')
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import lossFunction
import cv2

from localizationModel import MyNet
from localizationModel import SpNet
from localizationModel import NGNet
from dataLoader import localizerLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
from math import *

from torchvision import transforms

epochNum = 50
learningRate = 1e-4
threshold1 = 0.9
threshold2 = 0.7
threshold3 = 0.5
threshold4 = 0.3
threshold5 = 0.1
threshold6 = 1e-4
threshold7 = 0
#dirPath = 'DataSet/minusNTU'
#dirPath = 'DataSet/minus23'
#dirPath = 'DataSet/Cambridge/ShopFacade/train.txt'
#dirPath = 'DataSet/Cambridge/OldHospital/train.txt'
#testPath = 'DataSet/Cambridge/OldHospital/test.txt'
#testPath = 'DataSet/Cambridge/ShopFacade/test.txt'
dirPath = 'DataSet/Cambridge/OldHospital/train_dis.txt'
testPath = 'DataSet/Cambridge/OldHospital/test_dis.txt'

classLoss = 0
localLoss = 0

pthName = 'SP_local_NoCN_'

np.set_printoptions(suppress = True)

testLoss = nn.MSELoss()

def main():

        args = argParse()

        myNet = MyNet()
        spNet = SpNet()
        ngNet = NGNet()

        if args.train and args.sp and not args.ng:
                print ('SPNet')
                train(args, spNet)
        elif args.ng:
                print ('NGNet')
                train(args, ngNet)
        elif args.train:
                print ('MyNet')
                train(args, myNet)
        
        #test(args, myNet)


def argParse():

        parser = argparse.ArgumentParser()

        parser.add_argument('--train', action = 'store_true',
                                                help = 'Train Network')
        parser.add_argument('--test', action = 'store_true',
                                                help = 'Test Network')
        parser.add_argument('--sp', action = 'store_true',
                                                help = 'Whether use SPNet')
        parser.add_argument('--ng', action = 'store_true',
                                                help = 'No grouping')

        args = parser.parse_args()
        return args


def lossFunctionOri(input, target, quaternion, quaterniongt, tvec, tvecgt,\
        epoch):

        global classLoss, localLoss, epochNum

        classficationLoss = F.binary_cross_entropy(torch.sigmoid(input),\
                target)  
        loLoss = lossFunction.localizationLoss()
        localizationLoss = loLoss(quaternion, quaterniongt, tvec, tvecgt)
        #loss = classficationLoss
        #loss = classficationLoss + localizationLoss
        #loss = localizationLoss

        #loss = (classficationLoss * (epochNum - epoch) + localizationLoss *\
        #epoch * 0.1) / epochNum 
 
        loss = (classficationLoss * epochNum) + ((localizationLoss *\
        epoch * 0.01) / epochNum)
        classLoss += classficationLoss
        localLoss += localizationLoss
        
        return loss


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

def getH(objPts, imgPts):

    return cv2.findHomography(objPts, imgPts)

def matrix2Quaternion(m):

    #m = np.linalg.inv(m)

    q = np.zeros(4)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    
    if tr > 0:
        #print ('0')
        s = sqrt(1.0 + tr) * 2
        q[0] = 0.25 * s
        q[1] = (m[2, 1] - m[1, 2]) / s
        q[2] = (m[0, 2] - m[2, 0]) / s
        q[3] = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        #print ('1')
        s = sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q[0] = (m[2, 1] - m[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (m[1, 0] - m[0, 1]) / s
        q[3] = (m[0, 2] - m[2, 0]) / s 
    elif (m[1, 1] > m[2, 2]):
        #print ('2')
        
        s = sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q[0] = (m[0, 2] - m[2, 0]) / s
        q[1] = (m[1, 0] - m[0, 1]) / s
        q[2] = 0.25 * s
        q[3] = (m[2, 1] - m[1, 2]) / s
        '''
        k = 0.5 / sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q[0] = k * (m[1, 0] + m[0, 1])
        q[1] = 0.25 / k
        q[2] = k * (m[2, 1] + m[1, 2])
        q[3] = k * (m[2, 0] - m[0, 2])
        '''
    else:
        #print ('3')
        s = sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q[0] = (m[1, 0] - m[0, 1]) / s
        q[1] = (m[0, 2] - m[2, 0]) / s
        q[2] = (m[2, 1] - m[1, 2]) / s
        q[3] = 0.25 * s

    return q
       

def vector2Matrix(v):
    roll, pitch, yaw = v[0] * pi / 180, v[1] * pi / 180, v[2] * pi / 180
    
    x = np.array([[1, 0, 0],
                [0, cos(roll), -sin(roll)],
                [0, sin(roll), cos(roll)]])

    y = np.array([[cos(pitch), 0, sin(pitch)],
                [0, 1, 0],
                [-sin(pitch), 0, cos(pitch)]])

    z = np.array([[cos(yaw), -sin(yaw), 0],
                [sin(yaw), cos(yaw), 0],
                [0, 0, 1]])
    test = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    R = test.dot(x).dot(y).dot(z)
    return R

def quaternion2Matrix(q):

    w, x, y, z = q[0], q[1], q[2], q[3]

    m = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

    return m

def matrix2Vector(m):
    roll, pitch, yaw = atan2(m[2, 1], m[2, 2]), asin(m[2, 0]), atan2(m[1, 0], m[0, 0])
    x, y, z = roll * 180 / pi, pitch * 180 / pi, yaw * 180 / pi
    return np.array([x, y, z])
   
def train(args, myNet):
        outFile = open('./networkResult.txt', 'a')
        global epochNum, pthName
        
        bestAcc = 0

        resultFile = open('./result.txt', 'a')

        #lossF = lossFunction.Loss_classi()
        #lossF = lossFunctionOri()
        optimizer = optim.Adam(params = myNet.parameters(), lr = learningRate)
        dataSet = localizerLoader(dirPath)
        dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = False, num_workers = 1)

        # cuda & dataparallel?

        for epochIdx, epoch in enumerate(range(epochNum)):

                global localLoss, classLoss

                locallLoss = 0
                classLoss = 0
                batchNum = 0

                print\
                ('------------------------------------------------------------------------------')
                print (epoch + 1, 'th epoch, ', pthName, ', ',\
                        learningRate,', grouping, oldHos')
                print\
                ('------------------------------------------------------------------------------')

                correct1 = 0
                correct2 = 0
                correct3 = 0
                correct4 = 0
                correct5 = 0
                correct6 = 0
                correct7 = 0
                baselineCorrect = 0
                total = 0
                inlier = 0

                myNet.apply(weights_init)
                myNet.train()

                for batchIdx, batch in enumerate(dataLoader):

                        #print (batchIdx, 'th Batch')
                        (corre, label, ransacLabel, focalLength, quaterniongt,
                                tvecgt, distance) = batch
                        #tvecgt = np.reshape(tvecgt, (3, 1))

                        corre = corre.squeeze()
                        label = label.squeeze()
                        ransacLabel = ransacLabel.squeeze()
                        distance = distance.squeeze()
                        corre = Variable(corre.float(), requires_grad = True)
                        label = Variable(label.float(), requires_grad = False)
                        ransacLabel = Variable(ransacLabel.float(),\
                                requires_grad = False)
                        distance = Variable(distance.float(), requires_grad = True)
                        quaterniongt = Variable(quaterniongt.float(),\
                                requires_grad = True)
                        tvecgt = Variable(tvecgt.float(), requires_grad = True)
                        #distance = distance.expand(corre.size()[0], 1)
                        distance = torch.unsqueeze(distance, 1)

                        # distance here
                        #corre = torch.cat((corre, distance), 1)


                        #batch = np.expand_dims(batch, axis = 0)
                        #print (corre.size())
                        if args.train and args.sp:
                                outLabel, weight = myNet(corre)
                                #weight, outLabel = myNet(corre)
                        elif args.train:
                                outLabel = myNet(corre)
                        outLabel = outLabel.squeeze()
                        correMask = (outLabel > threshold7)
                        ransacMask = (label > 0)
                        #print (correMask.size(), corre.size())
                        #ransacCorre = corre[ransacMask]
                        corre = corre[correMask]
                        #print (corre.size())
                        intrinsicMatrix = np.array([[focalLength, 0.0, 0.0],\
                                                    [0.0, -focalLength, 0.0],\
                                                    [0.0, 0.0, 1.0]])
                        #ret, intrinsicMatrix, dist, r, t =\
                        #cv2.calibrateCamera(corre[:,2:], corre[:,:2], (1440,\
                        #    1080), intrinsicMatrix, None)
                        distCoeffs = np.zeros((5, 1))
                        corre = corre.data.numpy()
                        corre = np.expand_dims(corre, axis = 2)
                        #ransacCorre = np.expand_dims(ransacCorre, axis = 2)
                        #print (corre[:,2:,:].shape, corre[:,:2,:].shape)
                        objPts = np.ascontiguousarray(corre[:, 2:5, :])
                        imgPts = np.ascontiguousarray(corre[:, :2, :])
                        #ransacobjPts = np.ascontiguousarray(ransacCorre[:, 2:5, :])
                        #ransacimgPts = np.ascontiguousarray(ransacCorre[:, :2, :])
                        #print (corre.shape)
                        if len(corre) < 4:
                            #print ('too less pts')
                            #print ('gt ptnb: ', len(ransacCorre))
                            continue
                        ret, rvec, tvec = cv2.solvePnP(objPts, imgPts, intrinsicMatrix, distCoeffs)
                        #if len(ransacCorre) < 4:
                        #    continue
                        #rret, rrvec, rtvec = cv2.solvePnP(ransacobjPts,\
                        #        ransacimgPts, intrinsicMatrix, distCoeffs)
                        #print ('oriRV: ', np.reshape(rrvec, (1, 3)))
                        #print ('aftRV: ', matrix2Vector(vector2Matrix(rrvec)),'\n')
                        #rotationMatrix = cv2.Rodrigues(rvec)[0]
                        rotationMatrix = vector2Matrix(rvec)
                        #rrotationMatrix = vector2Matrix(rrvec)
                        newRvec = matrix2Vector(rotationMatrix)
                        #rnewRvec = matrix2Vector(rrotationMatrix)
                        quaternion =\
                        Variable(torch.from_numpy(matrix2Quaternion(rotationMatrix)).float(),\
                                requires_grad = True)
                        #rquaternion =\
                        #Variable(torch.from_numpy(matrix2Quaternion(rrotationMatrix)).float(),\
                        #        requires_grad = True)

                        quaternion = torch.reshape(quaternion, (1, 4))
                        #tvec = np.linalg.inv(rotationMatrix).dot(tvec)
                        #rtvec = -np.linalg.inv(rrotationMatrix).dot(rtvec)
                        #rtvec = rrotationMatrix.dot(rtvec)
                        #print (tvec.size)
                        #tvec = tvec.reshape(tvec, (1, 3))
                        #rtvec = np.reshape(rtvec, (1, 3))
                        gtRotationMatrix = quaternion2Matrix(quaterniongt[0])
                        rvecgt = matrix2Vector(gtRotationMatrix)
                        rvecgt = Variable(torch.from_numpy(rvecgt).float())
                        rvec = rvec.transpose()
                        rvec = rvec * 180 / pi
                        rvec[0, 1] *= -1
                        rvec = Variable(torch.from_numpy(rvec).float(),\
                                requires_grad = True)
                        #print ('rV: ', rrvec)
                        #print ('netrV:', rvec)
                        #print ('gtRV:', gtRotationVector, '\n')
                        #print ('M: ', rrotationMatrix)
                        #print ('invM', np.linalg.inv(rrotationMatrix))
                        #print ('gtM:', gtRotationMatrix, '\n')
                        #print ('r: ',rquaternion)
                        #print ('gtr: ', quaterniongt, '\n')
                        #print ('t: ', rtvec)
                        #print ('gtt: ', tvecgt, '\n')
                        #print ('-------------------------------')
                        #tvec = rotationMatrix.dot(tvec)
                        tvec = Variable(torch.from_numpy(tvec).float(),
                                requires_grad = True)
                        tvec = torch.reshape(tvec, (1, 3))
                        #loss = lossF(outLabel, label)
                        loss = lossFunctionOri(outLabel, label, rvec,\
                                rvecgt, tvec, tvecgt, epoch + 1)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # cal train acc and test

                        
                        for result, baseline, gt in zip(outLabel, ransacLabel, label):  
                                correct1 += 1 if result > threshold1 and gt == 1 else 0
                                correct1 += 1 if result < threshold1 and gt == 0 else 0
                                correct2 += 1 if result > threshold2 and gt == 1 else 0
                                correct2 += 1 if result < threshold2 and gt == 0 else 0
                                correct3 += 1 if result > threshold3 and gt == 1 else 0
                                correct3 += 1 if result < threshold3 and gt == 0 else 0
                                correct4 += 1 if result > threshold4 and gt == 1 else 0
                                correct4 += 1 if result < threshold4 and gt == 0 else 0
                                correct5 += 1 if result > threshold5 and gt == 1 else 0
                                correct5 += 1 if result < threshold5 and gt == 0 else 0
                                correct6 += 1 if result > threshold6 and gt == 1 else 0
                                correct6 += 1 if result < threshold6 and gt == 0 else 0
                                correct7 += 1 if result > threshold7 and gt == 1 else 0
                                correct7 += 1 if result < threshold7 and gt == 0 else 0
                                inlier += 1 if result > threshold7 else 0
                                baselineCorrect += 1 if baseline == gt else 0

                        batchNum += 1

                        total += len(label)
                        print (batchIdx, 'th Batch', file = outFile)
                        print (outLabel, file = outFile)

                acc1 = correct1 / total
                acc2 = correct2 / total
                acc3 = correct3 / total
                acc4 = correct4 / total
                acc5 = correct5 / total
                acc6 = correct6 / total
                acc7 = correct7 / total
                baselineAcc = baselineCorrect / total
                inlierP = inlier / total
                
                print ('\nbaseline: ', baselineAcc, '\n')
                print ('acc' , threshold1 , ': ', acc1, '\n')
                print ('acc' , threshold2 , ': ', acc2, '\n')
                print ('acc' , threshold3 , ': ', acc3, '\n')
                print ('acc' , threshold4 , ': ', acc4, '\n')
                print ('acc' , threshold5 , ': ', acc5, '\n')
                print ('acc' , threshold6 , ': ', acc6, '\n')
                print ('acc' , threshold7 , ': ', acc7, '\n')
                print ('inlier: ', inlierP)
                #print ('loss: ', loss, '\n')
                #print ('baseline: ', baselineAcc, '\n', file = resultFile)
                #print ('acc: ', acc, '\n\n', file = resultFile)

                testAcc1, testAcc2, testAcc3, testAcc4, testAcc5, testAcc6,\
                testAcc7, baseAcc, inlierP = test(args, myNet)

                print ('testBaseAcc: ', baseAcc)
                print ('testAcc1: ', testAcc1)
                print ('testAcc2: ', testAcc2)
                print ('testAcc3: ', testAcc3)
                print ('testAcc4: ', testAcc4)
                print ('testAcc5: ', testAcc5)
                print ('testAcc6: ', testAcc6)
                print ('testAcc7: ', testAcc7)
                print ('inlier: ', inlierP)

                torch.save(myNet.state_dict(), './pth/' + pthName +\
                        str(epochIdx) + 'epoch.pth')
                
                print ('BatchNum: ', batchNum)
                print ('classLoss: ', classLoss / batchNum)
                print ('localLoss: ', localLoss / batchNum)


def test(args, myNet):

        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        correct6 = 0
        correct7 = 0
        inlier = 0
        baselineCorrect = 0
        total = 0

        dataSet = localizerLoader(testPath)
        dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = False, num_workers = 1)

        myNet.eval()

        for batchIdx, batch in enumerate(dataLoader):

                (corre, label, ransacLabel, focalLength, quaternion, tvec,\
                        distance) = batch
                corre = corre.squeeze()
                label = label.squeeze()
                ransacLabel = ransacLabel.squeeze()
                distance = distance.squeeze()
                corre = Variable(corre.float(), requires_grad = False)
                label = Variable(label.float(), requires_grad = False)
                ransacLabel = Variable(ransacLabel.float(), requires_grad = False)
                distance = Variable(distance.float(), requires_grad = False)

                distance = torch.unsqueeze(distance, 1)
                
                # distance here
                #corre = torch.cat((corre, distance), 1)

                if args.train and args.sp:
                        outLabel, weight = myNet(corre)
                        #weight, outLabel = myNet(corre)
                elif args.train:
                        outLabel = myNet(corre)
                        outLabel = outLabel.squeeze()
                '''
                for result, baseline, gt in zip(outLabel, ransacLabel, label):  
                        #correct += 1 if result > threshold7 and gt == 1 else 0
                        if result > threshold7 and gt == 1:
                            correct += 1
                        #elif gt == 1:
                        #    print (result, gt, 'predict neg')
                        if result < threshold7 and gt == 0:
                            correct += 1
                        #elif gt == 0:
                        #    print (result, gt, 'predict pos')
                        #correct += 1 if result < threshold7 and gt == 0 else 0
                        baselineCorrect += 1 if baseline == gt else 0
                '''
                for result, baseline, gt in zip(outLabel, ransacLabel, label):  
                    correct1 += 1 if result > threshold1 and gt == 1 else 0
                    correct1 += 1 if result < threshold1 and gt == 0 else 0
                    correct2 += 1 if result > threshold2 and gt == 1 else 0
                    correct2 += 1 if result < threshold2 and gt == 0 else 0
                    correct3 += 1 if result > threshold3 and gt == 1 else 0
                    correct3 += 1 if result < threshold3 and gt == 0 else 0
                    correct4 += 1 if result > threshold4 and gt == 1 else 0
                    correct4 += 1 if result < threshold4 and gt == 0 else 0
                    correct5 += 1 if result > threshold5 and gt == 1 else 0
                    correct5 += 1 if result < threshold5 and gt == 0 else 0
                    correct6 += 1 if result > threshold6 and gt == 1 else 0
                    correct6 += 1 if result < threshold6 and gt == 0 else 0
                    correct7 += 1 if result > threshold7 and gt == 1 else 0
                    correct7 += 1 if result < threshold7 and gt == 0 else 0
                    baselineCorrect += 1 if baseline == gt else 0 
                    inlier += 1 if result > threshold7 else 0

                total += len(label)
        acc1 = correct1 / total
        acc2 = correct2 / total
        acc3 = correct3 / total
        acc4 = correct4 / total
        acc5 = correct5 / total
        acc6 = correct6 / total
        acc7 = correct7 / total
        #acc = correct / total
        baselineAcc = baselineCorrect / total

        return acc1, acc2, acc3, acc4, acc5, acc6, acc7, baselineAcc, inlier / total

if __name__ == '__main__':
        main()
