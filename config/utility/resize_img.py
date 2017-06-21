import cv2
import utility as ut
import os
from threading import Thread
from time import sleep

outputDir = "../../data/28/"
inputDir = "../../data/128/"

folders = os.listdir(inputDir)
print len(folders)

threadNum = 10

def convert(num):
	counter = 0
	print 'Thread: ', num
	for folder in folders:
		if ".DS_Store" not in folder:
			if os.path.exists(outputDir + folder) == False:
				os.mkdir(outputDir + folder)
			# 	continue
			# else:
			files = os.listdir(inputDir + folder)
			print len(files)
			for file in files:
				if ".DS_Store" not in file:
					# print 'Thread: ', num, inputDir + folder + "/" +file
					if os.path.exists(outputDir + folder + "/" +file) == False:
						img = cv2.imread(inputDir + folder + "/" +file, 1)
						if img != None:
							# image = cv2.resize(img, (224, 224))
							img = ut.resize(img)
							newImg, x, y = ut.scale(img, [], [],  imSize = 28)

							cv2.imwrite(outputDir + folder + "/" + file, newImg)
							# print 'Thread: ', num, "write to : ", outputDir + folder + "/" +file
					else:
						pass
						# print "skip: " + outputDir + folder + "/" +file
			counter += 1
			print counter

					# sleep(0.5)
# print newImg.shape

try:
	for i in range(threadNum):
		print(i)
		Thread(target = convert, args = (i,)).start()
		sleep(0.1)
except KeyboardInterrupt:
		sys.exit()
