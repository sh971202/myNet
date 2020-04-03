import argparse

def argParse():
	parser = argparse.ArgumentParser()
	parser.add_argument('txt', help = 'txt that would be transformed')

	args = parser.parse_args()
	return args

args = argParse()

txt = open(arg.txt, 'r')

for line in txt:
	line = line.replace('/', '_')