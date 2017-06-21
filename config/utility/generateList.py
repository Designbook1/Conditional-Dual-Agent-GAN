import os
import cv2

name = "10train"
# # name = "10test"
# name = "10_train_celeb_cropped"
# name = "10_test_celeb_cropped"
# name = "10test_Ext"
# name = "10train_Ext"

def writeToFile(content):
	with open(name + '.txt', 'a') as f:
	# with open('baseImage_224.txt', 'a') as f:
	# with open('GANExtNv_128.txt', 'a') as f:
		f.write(content)


with open(name + '.txt', 'w') as f:
	f.write("")


# inputDir = "/home/james/MS_GAN/data/" + name + "/"
# # inputDir = "/home/james/MS-Celeb-1M/lowshotImg_cropped_224/"
inputDir = "/home/james/MS_GAN/data/" + name + "/"


counter = 0
files = os.listdir(inputDir)
content = ""
print "len(files): ", len(files)

for index in range(len(files)):
	file = files[index]
	if ".DS_Store" not in file:
		# if "m." in file:
		pics = os.listdir(inputDir + file)
		print "len(pics): ", len(pics)
		# if len(pics) != 5:
			# print file
			# raise "debug"
		for pic in pics:
			if ".DS_Store" not in pic:
				path = (inputDir + file + "/" + pic)
				img = cv2.imread(path)
				if img != None:
					pass
				else:
					print line
					raise "debug"
				content += path + "\t" + str(index) + "\n"
				# content += path + " " + file + "\n"
				counter += 1
				# if counter % 100 == 0:
				print "counter: ", counter
writeToFile(content)
content = ""
# writeToFile(content)
# content = ""
