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

from distanceModel import MyNet
from distanceModel import SpNet
from distanceModel import NGNet
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
rloss = 0
tloss = 0

pthName = 'SP_class&local_distance_NoCN_'

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


def lossFunctionOri(input, target, rvec, rvecgt, tvec, tvecgt, epoch):

        global classLoss, localLoss, epochNum, rloss, tloss
        classficationLoss = F.binary_cross_entropy(torch.sigmoid(input),\
                target) 
        loLoss = lossFunction.localizationLoss()
        localizationLoss, rl, tl = loLoss(rvec, rvecgt, tvec, tvecgt)
        
        #loss = classficationLoss
        #loss = classficationLoss + localizationLoss
        #loss = localizationLoss

        loss = (classficationLoss * epochNum * 50 ) + ((localizationLoss *
        epoch * 0.1) / epochNum)

        classLoss += classficationLoss
        localLoss += localizationLoss
        rloss += rl
        tloss += tl
        
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

                global localLoss, classLoss, rloss, tloss

                classLoss = 0
                localLoss = 0
                rloss = 0
                tloss = 0
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
                        #print ('train Corre Size: ', corre.shape)
                        label = label.squeeze()
                        ransacLabel = ransacLabel.squeeze()
                        distance = distance.squeeze()
                        corre = Variable(corre.float(), requires_grad = True)
                        label = Variable(label.float())
                        ransacLabel = Variable(ransacLabel.float())
                        distance = Variable(distance.float(), requires_grad = True)
                        quaterniongt = Variable(quaterniongt.float(),\
                                requires_grad = False)
                        tvecgt = Variable(tvecgt.float(), requires_grad = False)
                        distance = torch.unsqueeze(distance, 1)

                        # distance here
                        corre = torch.cat((corre, distance), 1)

                        if args.train and args.sp:
                                outLabel, weight = myNet(corre)
                                #weight, outLabel = myNet(corre)
                        elif args.train:
                                outLabel = myNet(corre)
                        trueMask = (label > 0)
                        ransacMask = (ransacLabel > 0)
                        #print ('true mask : ', trueMask)
                        trueCorre = corre[trueMask]
                        ransacCorre = corre[ransacMask]
                        #print ('true mask size: ', trueMask.shape)
                        #print ('true corre size: ', trueCorre.shape)
                        #print ('ransac mask size: ', ransacMask.shape)
                        #print ('ransac corre size: ', ransacCorre.shape)
                        trueCorre = np.expand_dims((trueCorre.data.numpy()),\
                                axis = 2)
                        outLabel = outLabel.squeeze()
                        correMask = (outLabel > threshold7)
                        corre = corre[correMask]
                        #print ('corre mask size:', correMask.shape)
                        #print ('corre size: ', corre.shape, '\n')
                        corre = corre.data.numpy()
                        corre = np.expand_dims(corre, axis = 2)
                        #print (corre[:,2:,:].shape, corre[:,:2,:].shape)
                        #print (corre.shape)
                        

                        intrinsicMatrix = np.array([[focalLength, 0, 0],\
                                                    [0, -focalLength, 0],\
                                                    [0, 0, 1]])
                        distCoeffs = np.zeros((5, 1))
                        objPts = np.ascontiguousarray(corre[:, 2:5, :])
                        imgPts = np.ascontiguousarray(corre[:, :2, :])
                        trueObjPts = np.ascontiguousarray(trueCorre[:, 2:5, :])
                        trueImgPts = np.ascontiguousarray(trueCorre[:, :2, :])
                        if len(corre) < 4:
                            continue

                        ret, rvec, tvec = cv2.solvePnP(objPts, imgPts,\
                                intrinsicMatrix, distCoeffs)
                        #if len(trueCorre) < 4:
                        #    print ('true < 4')
                        #trueRet, trueRvec, trueTvec = cv2.solvePnP(trueObjPts,\
                        #        trueImgPts, intrinsicMatrix, distCoeffs)

                        rvecgt = matrix2Vector(quaternion2Matrix(quaterniongt[0]))
                        rvecgt = Variable(torch.from_numpy(rvecgt).float())
                        rvec = (rvec.transpose()) * 180 / pi
                        #rvec = rvec.transpose()
                        #rvec[0, 1] *= -1
                        rvec = rvec[0]
                        quaternion = matrix2Quaternion(vector2Matrix(rvec))
                        #print ('Q: ', quaternion)
                        #print ('Qgt: ', quaterniongt, '\n')
                        #rvec = Variable(torch.from_numpy(rvec).float(),\
                        #        requires_grad = True)
                        quaternion =\
                        Variable(torch.from_numpy(quaternion).float(),\
                                requires_grad = True)
                        tvec = Variable(torch.from_numpy(tvec).float(),\
                                requires_grad = True)
                        tvec = torch.reshape(tvec, (1, 3))

                        #trueRvec = (trueRvec.transpose()) * 180 / pi
                        #trueRvec[0, 1] *= -1

                        #print ('r: ', trueRvec)
                        #print ('rgt: ', rvecgt)
                        #print ('t: ', trueTvec)
                        #print ('tgt: ', tvecgt)
                        #print ('')

                        loss = lossFunctionOri(outLabel, label, quaternion,\
                                quaterniongt, tvec, tvecgt, epoch + 1)
                        
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

                                baselineCorrect += 1 if baseline == gt else 0
                                inlier += 1 if result > threshold7 else 0
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
                
                print ('\nbaseline: ', baselineAcc)
                print ('acc' , threshold1 , ': ', acc1)
                print ('acc' , threshold2 , ': ', acc2)
                print ('acc' , threshold3 , ': ', acc3)
                print ('acc' , threshold4 , ': ', acc4)
                print ('acc' , threshold5 , ': ', acc5)
                print ('acc' , threshold6 , ': ', acc6)
                print ('acc' , threshold7 , ': ', acc7)
                print ('inlier: ', inlierP, '\n')
                #print ('loss: ', loss, '\n')
                #print ('baseline: ', baselineAcc, '\n', file = resultFile)
                #print ('acc: ', acc, '\n\n', file = resultFile)

                testAcc1, testAcc2, testAcc3, testAcc4, testAcc5, testAcc6,\
                testAcc7, baseAcc, inlierP, rnl, tnl, rrnl, rtnl, plusrnl,\
                plustnl, testBatchNum = test(args, myNet)

                print ('testBaseAcc: ', baseAcc)
                print ('testAcc1: ', testAcc1)
                print ('testAcc2: ', testAcc2)
                print ('testAcc3: ', testAcc3)
                print ('testAcc4: ', testAcc4)
                print ('testAcc5: ', testAcc5)
                print ('testAcc6: ', testAcc6)
                print ('testAcc7: ', testAcc7, '\n')
                print ('inlier: ', inlierP, '\n')

                torch.save(myNet.state_dict(), './pth/' + pthName +\
                        str(epochIdx) + 'epoch.pth')
                
                print ('BatchNum: ', batchNum)
                print ('classLoss: ', classLoss / batchNum)
                print ('localLoss: ', localLoss / batchNum)
                print ('rLoss: ', rloss / batchNum)
                print ('tLoss: ', tloss / batchNum, '\n')

                print ('testBatchNum: ', testBatchNum)
                print ('testBase rLoss: ', rrnl / testBatchNum)
                print ('testBase tLoss: ', rtnl / testBatchNum)
                print ('test rLoss: ', rnl / testBatchNum)
                print ('test tLoss: ', tnl / testBatchNum)
                print ('test+RANSAC rLoss: ', plusrnl / testBatchNum)
                print ('test+RANSAC tLoss: ', plustnl / testBatchNum)




def test(args, myNet):

        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        correct6 = 0
        correct7 = 0
        inlier = 0
        rnl = 0
        tnl = 0
        rrnl = 0
        rtnl = 0
        plusrnl = 0
        plustnl = 0
        baselineCorrect = 0
        total = 0
        testBatchNum = 0


        dataSet = localizerLoader(testPath)
        dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = False, num_workers = 1)

        myNet.eval()

        for batchIdx, batch in enumerate(dataLoader):

                (corre, label, ransacLabel, focalLength, quaterniongt, tvecgt,\
                        distance) = batch
                corre = corre.squeeze()
                label = label.squeeze()
                ransacLabel = ransacLabel.squeeze()
                distance = distance.squeeze()
                corre = Variable(corre.float(), requires_grad = False)
                label = Variable(label.float(), requires_grad = False)
                ransacLabel = Variable(ransacLabel.float(), requires_grad = False)
                distance = Variable(distance.float(), requires_grad = False)
                quaterniongt = Variable(quaterniongt.float(), requires_grad =\
                        False)
                tvecgt = Variable(tvecgt.float(), requires_grad = False)
                distance = torch.unsqueeze(distance, 1)
                
                # distance here
                corre = torch.cat((corre, distance), 1)
                #outLabel = outLabel.squeeze()
                ransacLabel = ransacLabel.squeeze()
                ransacCorreMask = (ransacLabel > threshold7)
                ransacCorre = corre[ransacCorreMask]
                ransacCorre = ransacCorre.data.numpy()
                ransacCorre = np.expand_dims(ransacCorre, axis = 2)
                trueMask = (label > 0)
                trueCorre = corre[trueMask]
                trueCorre = np.expand_dims((trueCorre.data.numpy()),\
                        axis = 2)
                trueObjPts = np.ascontiguousarray(trueCorre[:, 2:5, :])
                trueImgPts = np.ascontiguousarray(trueCorre[:, :2, :])
                
                if args.train and args.sp:
                        outLabel, weight = myNet(corre)
                        #weight, outLabel = myNet(corre)
                elif args.train:
                        outLabel = myNet(corre)
                        outLabel = outLabel.squeeze()

                outLabel = outLabel.squeeze()
                correMask = (outLabel > threshold7)
                corre = corre[correMask]
                corre = corre.data.numpy()
                corre = np.expand_dims(corre, axis = 2)
                        
                intrinsicMatrix = np.array([[focalLength, 0, 0],\
                                        [0, -focalLength, 0],\
                                                    [0, 0, 1]])
                distCoeffs = np.zeros((5, 1))
                objPts = np.ascontiguousarray(corre[:, 2:5, :])
                imgPts = np.ascontiguousarray(corre[:, :2, :])
                ransacObjPts = np.ascontiguousarray(ransacCorre[:, 2:5, :])
                ransacImgPts = np.ascontiguousarray(ransacCorre[:, :2, :])
                if (len(corre) < 4) or (len(ransacCorre) < 4) :
                        continue

                ret, rvec, tvec = cv2.solvePnP(objPts, imgPts,\
                        intrinsicMatrix, distCoeffs)
                rret, rrvec, rtvec = cv2.solvePnP(ransacObjPts, ransacImgPts,\
                        intrinsicMatrix, distCoeffs)
                _, plusrvec, plustvec, plusinlier = cv2.solvePnPRansac(ransacObjPts, ransacImgPts,\
                        intrinsicMatrix, distCoeffs)
                if len(trueCorre) < 4:
                    print ('true < 4')
                    continue
                trueRet, trueRvec, trueTvec = cv2.solvePnP(trueObjPts,\
                        trueImgPts, intrinsicMatrix, distCoeffs)
                
                rvecgt = matrix2Vector(quaternion2Matrix(quaterniongt[0]))
                rvecgt = Variable(torch.from_numpy(rvecgt).float())
                rvec = (rvec.transpose()) * 180 / pi
                #rvec = rvec.transpose()
                #rvec[0, 1] *= -1
                rvec = rvec[0]
                rrvec = rrvec.transpose() * 180 / pi
                #rrvec = rrvec.transpose()
                #rrvec[0, 1] *= -1
                rrvec = rrvec[0]
                plusrvec = plusrvec.reshape(1, 3)
                plusrvec = plusrvec * 180 / pi
                #plusrvec[0, 1] *= -1
                plusrvec = plusrvec[0]
                trueRvec = trueRvec * 180 / pi
                trueRvec = trueRvec.transpose()
                #trueRvec[0, 1] *= -1
                trueRvec = trueRvec[0]
                quaternion = matrix2Quaternion(vector2Matrix(rvec))
                quaternion = Variable(torch.from_numpy(quaternion).float(),\
                        requires_grad = False)
                rquaternion = matrix2Quaternion(vector2Matrix(rrvec))
                rquaternion = Variable(torch.from_numpy(rquaternion).float(),\
                        requires_grad = False)
                plusquaternion = matrix2Quaternion(vector2Matrix(plusrvec))
                plusquaternion = Variable(torch.from_numpy(plusquaternion).float(),\
                        requires_grad = False)
                truequaternion = matrix2Quaternion(vector2Matrix(trueRvec))
                truequaternion = Variable(torch.from_numpy(truequaternion).float(),\
                        requires_grad = False)
                #quaterniongt =\
                #Variable(torch.from_numpy(quaterniongt).float(),\
                #        requires_grad = False)
                #rvec = Variable(torch.from_numpy(rvec).float(),\
                #                requires_grad = True)
                tvec = Variable(torch.from_numpy(tvec).float(),\
                                requires_grad = False)
                tvec = torch.reshape(tvec, (1, 3))
                rtvec = Variable(torch.from_numpy(rtvec).float(),\
                                requires_grad = False)
                rtvec = torch.reshape(rtvec, (1, 3))
                plustvec = Variable(torch.from_numpy(plustvec).float(),\
                                requires_grad = False)
                plustvec = torch.reshape(plustvec, (1, 3))

                #rvec = torch.div(rvec, (torch.sqrt(torch.sum(rvec ** 2))))
                #rvecgt = torch.div(rvecgt, (torch.sqrt(torch.sum(rvecgt ** 2))))
                #tvec = torch.div(tvec, (torch.sqrt(torch.sum(tvec ** 2))))
                #tvecgt = torch.div(tvecgt, (torch.sqrt(torch.sum(tvecgt ** 2))))
                #print (type(quaternion), type(quaterniongt))
                #rnl += torch.dist(rvec, rvecgt, 2)
                #print ('trueRv: ', trueRvec)
                #print ('gtRv: ', rvecgt)
                #print ('testRv: ', rvec)
                #print ('trueQ: ', truequaternion)
                #print ('gtQ: ', quaterniongt)
                #print ('testQ: ', quaternion, '\n')
                
                rnl += torch.dist(quaternion, quaterniongt, 2)
                tnl += torch.dist(tvec, tvecgt, 2)
                rrnl += torch.dist(rquaternion, quaterniongt, 2)
                rtnl += torch.dist(rtvec, tvecgt, 2)
                plusrnl += torch.dist(plusquaternion, quaterniongt, 2)
                plustnl += torch.dist(plustvec, tvecgt, 2)
                '''
                print ('test baseline r loss: ', torch.dist(rquaternion,\
                    quaterniongt, 2))
                print ('test baseline t loss: ', torch.dist(rtvec,\
                    tvecgt, 2))
                print ('test r loss: ', torch.dist(quaternion,\
                    quaterniongt, 2))
                print ('test t loss: ', torch.dist(tvec,\
                    tvecgt, 2))
                print ('test+RANSAC r loss: ', torch.dist(plusquaternion,\
                    quaterniongt, 2))
                print ('test+RANSAC t loss: ', torch.dist(plustvec,\
                    tvecgt, 2), '\n')
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
                testBatchNum += 1
        if total > 0:        
            acc1 = correct1 / total
            acc2 = correct2 / total
            acc3 = correct3 / total
            acc4 = correct4 / total
            acc5 = correct5 / total
            acc6 = correct6 / total
            acc7 = correct7 / total
            baselineAcc = baselineCorrect / total
            inlierP = inlier / total
        else:
            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc4 = 0
            acc5 = 0
            acc6 = 0
            acc7 = 0
            baselineAcc = 0
            inlierP = 0
            testBatchNum = -1

        return acc1, acc2, acc3, acc4, acc5, acc6, acc7, baselineAcc, inlierP,\
    rnl, tnl, rrnl, rtnl, plusrnl, plustnl, testBatchNum

if __name__ == '__main__':
        main()
