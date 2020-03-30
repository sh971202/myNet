import sys
import os
import fileinput
import argparse

def argParse():
	parser = argparse.ArgumentParser()
	parser.add_argument('dir', help = 'dir of list of test imgs')
	parser.add_argument('desDir', help = 'where test img will moved')

	args = parser.parse_args()
	return args

args = argParse()

testList = open(args.dir + 'dataset_test.txt')
for lineIdx, line in enumerate(testList):

	line = line.split()
	# should be [ImgName, 3 Position, 4 Rotation]
	if len(line) != 8:
		continue

	# cp testimg desdir
	cmd = 'cp ' + args.dir + line[0] + ' ' + args.desDir
	print (cmd)
	os.system(cmd)
