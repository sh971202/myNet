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

from model import MyNet
from model import SpNet
from model import NGNet
from dataLoader import localizerLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

epochNum = 50
learningRate = 1e-4
threshold1 = 0.9
threshold2 = 0.7
threshold3 = 0.5
threshold4 = 0.3
threshold5 = 0.1
threshold6 = 1e-4
threshold7 = 1e-5
#dirPath = 'DataSet/minusNTU'
#dirPath = 'DataSet/minus23'
dirPath = 'DataSet/Cambridge/ShopFacade'

def main():

	args = argParse()

	myNet = MyNet()
	spNet = SpNet()
	ngNet = NGNet()

	if args.train and args.sp:
		print ('SPNet')
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

	args = parser.parse_args()
	return args


def lossFunctionOri(input, target):

	return F.binary_cross_entropy(torch.sigmoid(input), target)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

def train(args, myNet):
	global epochNum
	# lack of optimizer

	resultFile = open('./result.txt', 'a')

	lossF = lossFunction.Loss_classi()
	#lossF = lossFunctionOri()
	optimizer = optim.Adam(params = myNet.parameters(), lr = learningRate)
	dataSet = localizerLoader(dirPath)
	dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = False, num_workers = 1)

	# cuda & dataparallel?

	for epoch in range(epochNum):

		print ('--------------------------------------------')
		print (epoch + 1, 'th epoch')
		print ('--------------------------------------------')

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
			(corre, label, ransacLabel) = batch
			corre = corre.squeeze()
			label = label.squeeze()
			ransacLabel = ransacLabel.squeeze()
			corre = Variable(corre.float())
			label = Variable(label.float())
			ransacLabel = Variable(ransacLabel.float())

			#batch = np.expand_dims(batch, axis = 0)

			optimizer.zero_grad()
			if args.train and args.sp:
				outLabel, weight = myNet(corre)
				#weight, outLabel = myNet(corre)
			elif args.train:
				outLabel = myNet(corre)
			outLabel = outLabel.squeeze()
				
			loss = lossF(outLabel, label)

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
			
			total += len(label)

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
		print ('loss: ', loss, '\n')
		#print ('baseline: ', baselineAcc, '\n', file = resultFile)
		#print ('acc: ', acc, '\n\n', file = resultFile)

def test(args, myNet):

	return

if __name__ == '__main__':
	main()
