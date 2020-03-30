import argparse

from os import listdir
from os.path import isfile, join
from PIL import Image

def argParse():
	parser = argparse.ArgumentParser()
	parser.add_argument('dir', help = 'dir of png imgs')

	args = parser.parse_args()
	return args


args = argParse()
if args.dir[-1] != '/':
	args.dir += '/'
files = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]

for file in files:

	if file[-4:] != '.png':
		continue

	png = Image.open(args.dir + file)
	jpg = png.convert('RGB')

	print ('Converting ' + file)

	imgName = args.dir + file.split('.')[0] + '.jpg'
	jpg.save(imgName)


