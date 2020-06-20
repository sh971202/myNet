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

from model128 import MyNet
from model128 import SpNet
from model128 import NGNet
from dataLoader128 import localizerLoader
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
dirPath = 'DataSet/Cambridge/OldHospital/128train.txt'
testPath = 'DataSet/Cambridge/OldHospital/128test.txt'

classLoss = 0

pthName = 'SP_class_Nodistance_NoCN_128_'

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


def lossFunctionOri(input, target, epoch):

        global classLoss, localLoss, epochNum

        classficationLoss = F.binary_cross_entropy(torch.sigmoid(input),\
                target)  
        loss = classficationLoss
        #loss = classficationLoss + localizationLoss
        #loss = localizationLoss

        #loss = (classficationLoss * (epochNum - epoch) + localizationLoss *
        #epoch * 0.1) / epochNum 

        classLoss += classficationLoss
        
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

                myNet.apply(weights_init)
                myNet.train()

                for batchIdx, batch in enumerate(dataLoader):

                        #print (batchIdx, 'th Batch')
                        (corre, label, ransacLabel, focalLength, quaterniongt,
                                tvecgt, des1, des2) = batch
                        #tvecgt = np.reshape(tvecgt, (3, 1))

                        corre = corre.squeeze()
                        label = label.squeeze()
                        ransacLabel = ransacLabel.squeeze()
                        #distance = distance.squeeze()
                        des1 = des1.squeeze()
                        des2 = des2.squeeze()
                        corre = Variable(corre.float())
                        label = Variable(label.float())
                        ransacLabel = Variable(ransacLabel.float())
                        #distance = Variable(distance.float())
                        des1 = Variable(des1.float())
                        des2 = Variable(des2.float())
                        quaterniongt = Variable(quaterniongt.float(),\
                                requires_grad = True)
                        tvecgt = Variable(tvecgt.float(), requires_grad = True)
                        #distance = torch.unsqueeze(distance, 1)
                        #des1 = torch.unsqueeze(des1, 1)
                        #des2 = torch.unsqueeze(des2, 1)

                        # distance here
                        #corre = torch.cat((corre, distance), 1)

                        corre = torch.cat((corre, des1), 1)
                        corre = torch.cat((corre, des2), 1)

                        if args.train and args.sp:
                                outLabel, weight = myNet(corre)
                                #weight, outLabel = myNet(corre)
                        elif args.train:
                                outLabel = myNet(corre)
                        outLabel = outLabel.squeeze()
                        correMask = (outLabel > threshold7)
                        corre = corre[correMask]
                        corre = corre.data.numpy()
                        corre = np.expand_dims(corre, axis = 2)
                        #print (corre[:,2:,:].shape, corre[:,:2,:].shape)
                        #print (corre.shape)
                 
                        loss = lossFunctionOri(outLabel, label, epoch + 1)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # cal train acc and test

                        myNet.eval()
                        
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
                
                print ('\nbaseline: ', baselineAcc, '\n')
                print ('acc' , threshold1 , ': ', acc1, '\n')
                print ('acc' , threshold2 , ': ', acc2, '\n')
                print ('acc' , threshold3 , ': ', acc3, '\n')
                print ('acc' , threshold4 , ': ', acc4, '\n')
                print ('acc' , threshold5 , ': ', acc5, '\n')
                print ('acc' , threshold6 , ': ', acc6, '\n')
                print ('acc' , threshold7 , ': ', acc7, '\n')
                #print ('loss: ', loss, '\n')
                #print ('baseline: ', baselineAcc, '\n', file = resultFile)
                #print ('acc: ', acc, '\n\n', file = resultFile)

                testAcc1, testAcc2, testAcc3, testAcc4, testAcc5, testAcc6, testAcc7, baseAcc = test(args, myNet)

                print ('testBaseAcc: ', baseAcc)
                print ('testAcc1: ', testAcc1)
                print ('testAcc2: ', testAcc2)
                print ('testAcc3: ', testAcc3)
                print ('testAcc4: ', testAcc4)
                print ('testAcc5: ', testAcc5)
                print ('testAcc6: ', testAcc6)
                print ('testAcc7: ', testAcc7)

                torch.save(myNet.state_dict(), './pth/' + pthName +\
                        str(epochIdx) + 'epoch.pth')
                
                print ('BatchNum: ', batchNum)
                print ('classLoss: ', classLoss / batchNum)


def test(args, myNet):

        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        correct6 = 0
        correct7 = 0

        baselineCorrect = 0
        total = 0

        dataSet = localizerLoader(testPath)
        dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = False, num_workers = 1)

        myNet.eval()

        for batchIdx, batch in enumerate(dataLoader):

                (corre, label, ransacLabel, focalLength, quaternion, tvec,\
                        des1, des2) = batch
                corre = corre.squeeze()
                label = label.squeeze()
                ransacLabel = ransacLabel.squeeze()
                #distance = distance.squeeze()
                des1 = des1.squeeze()
                des2 = des2.squeeze()
                corre = Variable(corre.float())
                label = Variable(label.float())
                ransacLabel = Variable(ransacLabel.float())
                #distance = Variable(distance.float())
                des1 = Variable(des1.float())
                des2 = Variable(des2.float())

                #distance = torch.unsqueeze(distance, 1)
                #des1 = torch.unsqueeze(des1, 1)
                #des2 = torch.unsqueeze(des2, 1)
                
                # distance here
                #corre = torch.cat((corre, distance), 1)

                corre = torch.cat((corre, des1), 1)
                corre = torch.cat((corre, des2), 1)

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

        return acc1, acc2, acc3, acc4, acc5, acc6, acc7, baselineAcc

if __name__ == '__main__':
        main()
