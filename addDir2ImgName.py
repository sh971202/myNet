import argparse
import os
import sys

from os.path import isfile, join
from os import listdir

def argParse():
	parser = argparse.ArgumentParser()
	parser.add_argument('dir', help = 'dir of jpgs')

	args = parser.parse_args()
	return args

args = argParse()
args.dir += '' if args.dir[-1] == '/' else '/'

files = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]

for file in files:

	if file[-4:] != '.jpg':
		continue

	# mv file dir_file
	cmd = 'mv ' + args.dir + file + ' ' + args.dir[:-1] + '_' + file 
	print (cmd)
	os.system(cmd)
