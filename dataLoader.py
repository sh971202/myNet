import sys
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader

class localizerLoader(Dataset):

	def __init__(self, dirPath):

		self.dirPath = dirPath

		count = -1
		for count, line in enumerate(open(dirPath + '/train.txt', 'rU')):
			pass
		count += 1

		self.data = np.empty((count, 5))
		self.label = np.empty((count, 1))
		self.ransacLabel = np.empty((count, 1))
		self.dic = {}
		self.nextDic = {}
		self.imageIDs = []

		imageID = ''
		npID = 0
		d = open(dirPath + '/train.txt', 'r')
		for line in d.readlines():
			corre = line.split()
			# corre : imageID, x, y, x, y, z, label, ransacLabel
			self.data[npID] = corre[1:-2]
			self.label[npID] = 1 if (corre[-2] == 'True') else 0
			self.ransacLabel[npID] = 1 if (corre[-1] == 'True') else 0
			if imageID != corre[0]: # new image
				self.imageIDs.append(imageID)
				self.nextDic.update({imageID: npID})
				imageID = corre[0]
				self.dic.update({imageID: npID})
			npID += 1

		self.imageIDs.append(imageID)
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

		return corre, label, ransacLabel




def main():

	return


if __name__ == '__main__':

	main()
