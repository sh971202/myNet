import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from sklearn.metrics.pairwise import euclidean_distances

def fullyConnected(in_channel, out_channel):
	return nn.Linear(in_channel, out_channel)
def con1x3(in_channel, out_channel):
	return nn.Conv2d(in_channel, out_channel, kernel_size = (3, 1))

class SpNet(nn.Module):
	def __init__(self):
		super(SpNet, self).__init__()

		self.fc = fullyConnected(5, 32)

		# Grouping 32 -> 32x8

		self.res1 = SpResNetBlock(32, 8)
		self.res2 = SpResNetBlock(32, 64)
		self.res3 = SpResNetBlock(64, 64)
		self.res4 = SpResNetBlock(64, 128)
		self.res5 = SpResNetBlock(128, 128)
		self.res6 = SpResNetBlock(128, 256)
		self.res7 = SpResNetBlock(256, 1)

	def forward(self, x):

		#print (x)
		#print (x.size())
		out = self.fc(x)
		out = torch.unsqueeze(out, 1)
		# out = 126 x 1 x 32

		# Grouping here
		dis = torch.from_numpy(euclidean_distances(x, x))
		sortIdx = torch.argsort(dis, dim = 1)

		#print (out.size())
		buf = out
		out = out.expand(126, 8, 32)
		for correIdx, corre in enumerate(buf):
			for idx in range(7):
				print (corre.size(), buf[sortIdx[correIdx][idx]].size())
				corre = torch.cat((corre, buf[sortIdx[correIdx][idx]]), 0)
				print (corre.size())
			out[correIdx] = corre

		print (out.size())
		#out = torch.unsqueeze(out, 0)
		out = out.permute(1, 2, 0)
		print (out.size())
		#print (out.size())
		out = self.res1(out)
		out = self.res2(out)
		out = self.res3(out)
		out = self.res4(out)
		out = self.res5(out)
		out = self.res6(out)
		out = self.res7(out)

		return out


class SpResNetBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(SpResNetBlock, self).__init__()

		self.conv = con1x3(in_channel, out_channel)
		self.inorm = nn.InstanceNorm1d(out_channel)
		self.bnorm = nn.BatchNorm1d(out_channel)
		
	def forward(self, x):

		res = x

		out = self.conv(x)
		out = self.inorm(out)
		out = self.bnorm(out)

		out += res
		return out


class MyNet(nn.Module):

	def __init__(self):

		super(MyNet, self).__init__()

		self.firstP = fullyConnected(5, 128)
		self.fc1 = fullyConnected(5, 16)
		self.fc2 = fullyConnected(16, 64)
		self.fc3 = fullyConnected(64, 128)
		
		self.res1 = ResNetBlock()
		self.res2 = ResNetBlock()
		self.res3 = ResNetBlock()
		self.res4 = ResNetBlock()
		self.res5 = ResNetBlock()
		self.res6 = ResNetBlock()
		self.res7 = ResNetBlock()
		self.res8 = ResNetBlock()
		self.res9 = ResNetBlock()
		self.res10 = ResNetBlock()
		self.res11 = ResNetBlock()
		self.res12 = ResNetBlock()

		self.finalP = fullyConnected(128, 1)
		self.testConv = nn.Conv2d(1, 1, 1)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, x):

		out = self.firstP(x)
		#out = self.fc1(x)
		#out = self.fc2(out)
		#out = self.fc3(out)

		out = self.res1(out)
		out = self.res2(out)
		out = self.res3(out)
		out = self.res4(out)
		out = self.res5(out)
		out = self.res6(out)
		out = self.res7(out)
		out = self.res8(out)
		out = self.res9(out)
		out = self.res10(out)
		out = self.res11(out)
		out = self.res12(out)

		out = self.finalP(out)
		out = self.testConv(out)
		out = self.relu(out)
		out = self.tanh(out)

		return out


class ResNetBlock(nn.Module):

	def __init__(self):

		super(ResNetBlock, self).__init__()

		self.P1 = fullyConnected(128, 128)
		self.batchNorm1 = nn.BatchNorm1d(128)
		self.relu1 = nn.ReLU()

		self.P2 = fullyConnected(128, 128)
		self.batchNorm2 = nn.BatchNorm1d(128)
		self.relu2 = nn.ReLU()


	def forward(self, x):

		def contextNormalization(x):
			#  x : N x 128
			#  mean : 128

			mean = torch.mean(x, 0)
			
			diff = x
			for d in diff:
				d -= mean
		
			#  va : 128
			va = torch.var(diff, 0)
			va = torch.sqrt( torch.div(va, x.size()[0] ) )

			x = torch.div(diff, va)

			return x


		res = x
		out = self.P1(x)
		contextNormalization(out)
		out = self.batchNorm1(out)
		out = self.relu1(out)

		out = self.P2(out)
		contextNormalization(out)
		out = self.batchNorm2(out)
		out = self.relu2(out)

		out += res

		return out

