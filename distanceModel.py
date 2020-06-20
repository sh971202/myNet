import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics.pairwise import euclidean_distances

def fullyConnected(in_channel, out_channel):
        return nn.Linear(in_channel, out_channel)
def con1x3(in_channel, out_channel):
        return nn.Conv2d(in_channel, out_channel, kernel_size = (1, 3))
def con2x3(in_channel, out_channel):
        return nn.Conv2d(in_channel, out_channel, kernel_size = (2, 3))

def pairwiseDistances(x, y = None):
    '''
    Input : x is a Nxd Matrix
            y is a Mxd Matrix
    Output : a NxM matrix where dist[i, j] is the square norm between
            x[i,:] and y[j:], if y is not given use 'y = x'
    '''

    xNorm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        yt = torch.transpose(y, 0, 1)
        yNorm = (y**2).sum(1).view(1, -1)
    else:
        yt = torch.transpose(x, 0, 1)
        yNorm = (x**2).sum(1).view(1, -1)

    dist = xNorm + yNorm - 2.0 * torch.mm(x, yt)
    return torch.clamp(dist, 0.0, np.inf)




class NGNet(nn.Module):
        def __init__(self):
                super(NGNet, self).__init__()

                self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, (1, 5)),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                        )
                self.conv2 = nn.Sequential(
                        nn.Conv2d(256, 256, (1, 1)),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )
                self.finalConv = nn.Sequential(
                        nn.Conv2d(256, 1, (1, 1)),
                        )
                self.layer = self.makeLayers(NGNetResNetBlock)


        def makeLayers(self, NGNetResNetBlock):

                layers = []
                layers.append(NGNetResNetBlock(32, 32))
                layers.append(NGNetResNetBlock(32, 32))
                layers.append(NGNetResNetBlock(32, 64, pre = True))
                layers.append(NGNetResNetBlock(64, 64))
                layers.append(NGNetResNetBlock(64, 128, pre = True))
                layers.append(NGNetResNetBlock(128, 128))
                layers.append(NGNetResNetBlock(128, 256, pre = True))
                layers.append(NGNetResNetBlock(256, 256))

                return nn.Sequential(*layers)


        def forward(self, x):
                
                x = torch.unsqueeze(x, 1)
                x = torch.unsqueeze(x, 1)
                out = self.conv1(x)
                #print ('After conv1, ', x.size())
                out = self.layer(out)
                #print ('layer F')
                out = self.conv2(out)
                out = self.finalConv(out)
                out = out.view(out.size(0), -1)
                w = torch.tanh(out)
                w = F.relu(w)

                return out ,w


class NGNetResNetBlock(nn.Module):
        def __init__(self, in_channel, out_channel, stride = 1, pre = False):
                super(NGNetResNetBlock, self).__init__()
                self.pre = pre
                self.right = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1), stride = (1, stride)),
            nn.BatchNorm2d(out_channel)
                )

                self.left = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, (1, 1), stride = (1, stride)),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(inplace = True),
                        nn.Conv2d(out_channel, out_channel, (1, 1), stride = 1),
                        nn.BatchNorm2d(out_channel),
                        )

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

                res = self.right(x) if self.pre is True else x
                out = self.left(x)
                out = out + res
                #contextNormalization(out)
                return F.relu(out)



class SpNet(nn.Module):
        def __init__(self):
                super(SpNet, self).__init__()

                self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, (1, 6)),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        )
                self.conv2 = nn.Sequential(
                        nn.Conv2d(256, 256, (1, 1)),
                        #nn.InstanceNorm2d(256),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        )
                self.finalConv = nn.Sequential(
                        nn.Conv2d(256, 1, (1, 1)),
                        )
                self.layer = self.makeLayers(SpResNetBlock)

        def makeLayers(self, SpResNetBlock):
                layers = []
                layers.append(SpResNetBlock(32, 32, pre = False))
                layers.append(SpResNetBlock(32, 32, pre = False))
                layers.append(SpResNetBlock(32, 64, stride = 2, pre = True))
                layers.append(SpResNetBlock(64, 64, pre = False))
                layers.append(SpResNetBlock(64, 128, stride = 2, pre = True))
                layers.append(SpResNetBlock(128, 128, pre = False))
                layers.append(SpResNetBlock(128, 256, stride = 2, pre = True))
                layers.append(SpResNetBlock(256, 256, pre = False))

                return nn.Sequential(*layers)

        def forward(self, x):
                #print (x)
                #c = x.detach().numpy()
                #dis = torch.from_numpy(euclidean_distances(c, c))
                disMatrix = pairwiseDistances(x)

                x = torch.unsqueeze(x, 1)
                x = torch.unsqueeze(x, 1)
                #print (x.size())
                out = self.conv1(x)
                #print (out.size())
                # grouping

                #dis = torch.from_numpy(euclidean_distances(x, x))
                sortIdx = torch.argsort(disMatrix, dim = 1)
                
                buf = out
                out = out.expand(x.size()[0], 32, 1, 8)
                for correIdx, corre in enumerate(buf):
                        for idx in range(7):
                                #print (corre.size(), buf[sortIdx[correIdx][idx]].size())
                                ''' idx + 1 because dont want x=y '''
                                corre = torch.cat((corre,\
                                    buf[sortIdx[correIdx][idx+1]]), 2)
                                #print (buf[sortIdx[correIdx][idx]])
                                #print (corre.size())
                        out[correIdx] = corre

                out = self.layer(out)
                out = self.conv2(out)
                out = self.finalConv(out)
                out = out.view(out.size(0), -1)
                w = torch.tanh(out)
                w = F.relu(w)

                return out, w


class SpResNetBlock(nn.Module):
        def __init__(self, in_channel, out_channel, stride = 1, pre = False):
                super(SpResNetBlock, self).__init__()
                self.pre = pre
                self.right = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, (1, 3), stride = (1, stride), padding = (0, 1)),
                        nn.BatchNorm2d(out_channel)
                        )
                self.left = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, (1, 3), stride = (1, stride), padding = (0, 1)),
                        #nn.InstanceNorm2d(out_channel),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(inplace = True),
                        nn.Conv2d(out_channel, out_channel, (1, 3), stride = 1, padding = (0, 1)),
                        #nn.InstanceNorm2d(out_channel),
                        nn.BatchNorm2d(out_channel)
                        )

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

                identity = self.right(x) if self.pre is True else x
                out = self.left(x)
                out = out + identity
                #contextNormalization(out)
                return F.relu(out)


class FinalResNetBlock(nn.Module):
        def __init__(self, in_channel, out_channel):
                super(FinalResNetBlock, self).__init__()

                self.conv = con1x3(in_channel, out_channel)
                self.inorm = nn.InstanceNorm2d(out_channel)
                self.bnorm = nn.BatchNorm2d(out_channel)
                self.relu = nn.ReLU()

                #self.conv2 = con1x3(out_channel, out_channel)
                self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = (1, 2))
                self.inorm2 = nn.InstanceNorm2d(out_channel)
                self.bnorm2 = nn.BatchNorm2d(out_channel)
                self.relu2 = nn.ReLU()
                
        def forward(self, x):

                res = x
                out = self.conv(x)
                #out = self.inorm(out)
                out = self.bnorm(out)
                out = self.relu(out)

                out = self.conv2(out)
                #out = self.inorm2(out)
                out = self.bnorm2(out)
                out = self.relu2(out)

                #out += res
                return out


class MyNet(nn.Module):

        def __init__(self):

                super(MyNet, self).__init__()

                self.firstP = fullyConnected(6, 128)
                self.fc1 = fullyConnected(6, 16)
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
                #out = self.testConv(out)
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

