import os
from time import sleep
import sys
import cv2
import utility as ut
debug = False
# debug = True

def augmentation(imgPath, name):
	# generateFunc = ["original", "mirror", "rotate", "translate", "brightnessAndContrast"]
	# generateFunc = ["mirror", "rotate", "translate", "brightnessAndContrast", "blur"]
	generateFunc = ["mirror", "brightnessAndContrast", "blur"]

	if debug:
		print "imgPath + name: ", imgPath + name

	img = cv2.imread(imgPath + name)
	x, y = [], []

	img = ut.resize(img)
	newImg, x, y = ut.scale(img, [], [],  imSize = 224)
	print "name: ", name
	print "type(img): ", type(img)
	print "img.shape: ", img.shape
	if img != None:
		if debug:
			print "FIND image: ", imgPath + name
			print "img.shape: ", img.shape
		derivateNum = len(generateFunc)
		for index in range(derivateNum):

			(w, h, _) = img.shape
			# method = random.choice(generateFunc)
			method = generateFunc[index]

			# if index == 0:
			# 	method = "original"

			# if method == "resize":
			#     newImg, newX, newY = ut.resize(img, x, y, xMaxBound = w, yMaxBound = h, random = True)
			if method == "rotate":
				newImg, newX, newY = ut.rotate(img, x, y, w = w, h = h)
			elif method == "mirror":
				newImg, newX, newY = ut.mirror(img, x, y, w = w, h = h)
			elif method == "translate":
				newImg, newX, newY = ut.translate(img, x, y, w = w, h = h)
			elif method == "brightnessAndContrast":
				newImg, newX, newY = ut.contrastBrightess(img, x, y)
			elif method == "original":
				newImg, newX, newY = img, x, y
			elif method == "blur":
				newImg, newX, newY  = blur = cv2.blur(img,(5,5)), x, y
				# newImg, newX, newY = ut.scale(img, x, y, imSize = imSize)
			# elif method == "scale":
			# 	newImg, newX, newY = ut.scale(img, x, y)
			else:
				raise "not existing function"

			if debug:
				print "name: ", name
				print "index: ", index
				print "newImg.shape: ", newImg.shape
				print "method: ", method
				cv2.imshow("img", img)
				cv2.imshow("newImg", newImg)
				cv2.waitKey(0)

			cv2.imwrite(path + name.replace(".jpg", "") + "_" + str(index) + ".jpg", newImg)
			print "saving to ............"
			print path + name.replace(".jpg", "") + "_" + str(index) + ".jpg"

# name = "10train"
# inputDir = "../../data/" + name + "/"
# input_dir = "/Users/zhecanwang/Project/MS_GAN/data/10_train_celeb_cropped/"
input_dir = '/home/james/MS_GAN/data/10train/'

counter = 0
folders = os.listdir(input_dir)
print "len(folders): ", len(folders)

for index in range(len(folders)):
	folder = folders[index]
	print "folder: ", folder
	if ".DS_Store" not in folder:
		# if "m." in folder:
		pics = os.listdir(input_dir + folder)
		print "len(pics): ", len(pics)
		for pic in pics:
			if ".DS_Store" not in pic:
				path = (input_dir + folder + "/")
				augmentation(path, pic)
				counter += 1
				if counter % 100 == 0:
					print "counter: ", counter
		# 		break
		# 		raise "debug"
		# break
		# raise "debug"
	else:
		pass
