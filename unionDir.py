import argparse

def argParse():
	parser = argparse.ArgumentParser()
	parser.add_argument('txt', help = 'txt that would be transformed')

	args = parser.parse_args()
	return args

args = argParse()

txt = open(args.txt, 'r')
outfile = open('./newTxt', 'a')

for line in txt:
	line = line.replace('/', '_')
	print (line, file = outfile, end = '')