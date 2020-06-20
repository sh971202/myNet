import sys
import os
import numpy as np

from random import sample
from torch.utils.data import Dataset, DataLoader

class localizerLoader(Dataset):

        def __init__(self, dirPath):

                self.dirPath = dirPath

                count = -1
                for count, line in enumerate(open(dirPath, 'rU')):
                        pass
                count += 1

                self.data = np.empty((count, 5))
                self.label = np.empty((count, 1))
                self.ransacLabel = np.empty((count, 1))
                #self.distance = np.empty((count, 1))
                self.des1 = np.empty((count, 128))
                self.des2 = np.empty((count, 128))
                self.dic = {}
                self.nextDic = {}
                self.imageIDs = []
                self.focalLength = []
                self.quaternion = []
                self.tvec = []
                

                imageID = ''
                npID = 0
                d = open(dirPath, 'r')
                for line in d.readlines():
                        corre = np.array(line.split())
                        # corre : imageID, x, y, x, y, z, label, ransacLabel,
                        # focalLength, quaternion, tvec, distance
                        self.data[npID] = corre[1:6]
                        self.label[npID] = 1 if (corre[6] == 'True') else 0
                        self.ransacLabel[npID] = 1 if (corre[7] == 'True') else 0
                        #self.distance[npID] = corre[16]
                        self.des1[npID] = corre[16:144]
                        self.des2[npID] = corre[144:272]
                        if imageID != corre[0]: # new image
                                self.imageIDs.append(imageID)
                                self.focalLength.append(float(corre[8]))
                                #print (corre)
                                self.quaternion.append(corre[9:13].astype(float))
                                self.tvec.append(corre[13:16].astype(float))
                                self.nextDic.update({imageID: npID})
                                imageID = corre[0]
                                self.dic.update({imageID: npID})
                        npID += 1

                self.imageIDs.append(imageID)
                self.focalLength.append(float(corre[8]))
                self.quaternion.append(corre[9:13].astype(float))
                self.tvec.append(corre[13:16].astype(float))
                self.imageIDs.remove('')
                del self.nextDic['']
                # Last image to the end of file
                self.nextDic.update({imageID: count})
                # shape = n, 6
                # imageID, x, y, x, y, z
                self.data = self.data.astype(np.float32)

                print("Get %d Corres" % count)

        # Read Groundtruth file
                '''
        gt = open(dirPath + '/groundtruth.txt', 'r')
        for lineCount, line in enumerate(gt.readlines()):
            if lineCount == 0:
                self.imageNum = int(line)
        '''

        def __len__(self):
                '''
                count = -1
                for count, line in enumerate(open(self.dirPath, 'rU')):
                        pass
                
                return count + 1
                '''

                return len(self.dic)

        def __getitem__(self, index):

                #index = str(index + 289)
                #print (self.imageIDs)
                imageID = self.imageIDs[index]
                corre = self.data[ self.dic[imageID] : self.nextDic[imageID] ]
                label = self.label[ self.dic[imageID] : self.nextDic[imageID] ]
                ransacLabel = self.ransacLabel[ self.dic[imageID] : self.nextDic[imageID] ]
                des1 = self.des1[ self.dic[imageID] : self.nextDic[imageID] ]
                des2 = self.des2[ self.dic[imageID] : self.nextDic[imageID] ]
                #distance = self.distance[ self.dic[imageID] :\
                #        self.nextDic[imageID] ]
                f = self.focalLength[index]
                r = self.quaternion[index]
                t = self.tvec[index]
                #return corre, label, ransacLabel, f, r, t, distance

                retCorre = []
                retLabel = []
                retRan = []
                
                for idx in sample(range(self.dic[imageID], self.nextDic[imageID]), self.nextDic[imageID] - self.dic[imageID]):
                   retCorre.append(self.data[idx])
                   retLabel.append(self.label[idx])
                   retRan.append(self.ransacLabel[idx]) 

                retCorre = np.array(retCorre)
                retLabel = np.array(retLabel)
                retRan = np.array(retRan)

                return corre, label, ransacLabel, f, r, t, des1, des2
                #return retCorre, retLabel, retRan, f, r, t, distance

def main():

        return


if __name__ == '__main__':

        main()
