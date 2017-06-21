import urllib
import os
from threading import Thread
from time import sleep
import urllib2
import sys
import cv2
import utility as ut

debug = False

inputDir = "/Users/zhecanwang/Project/MS_GAN/data/10_train_celeb/"
output_dir = "/Users/zhecanwang/Project/MS_GAN/data/10_train_celeb_cropped/"

folders = os.listdir(inputDir)
print "len(folders): ", len(folders)

for index in range(len(folders)):
	folder = folders[index]
	# folder = "angelina jolie"
	print "folder: ", folder
	path = inputDir + folder

	if ".DS_Store" not in folder:
		if os.path.exists(output_dir + folder) == False:
				os.mkdir(output_dir + folder)
		# file_name = "info.txt"

		file_name = "/filelist_LBP.txt"
		with open(path + file_name, 'r') as f:
			lines = f.readlines()

		for index in range(len(lines)):
			print "index: ", index
			line = lines[index].replace("\r\n", "")

			line_list = line.split("\t")
			print "line_list: ", line_list
			img_name = line_list[0]

			# if img_name == "7.jpg":
			# if img_name == "1726.jpg":
			# 	debug = True

			print img_name
			img = cv2.imread(path + "/" + img_name)

			pts = line_list[1:5]
			h, w = float(line_list[5]), float(line_list[6])
			print "pts: ", pts
			print "w, h: ", w, h
			x1, y1, x2, y2 = [int(i) for i in pts]
			x1, x2 = x1/w, x2/w
			y1, y2 = y1/h, y2/h

			w, h, _ = img.shape

			x1, x2 = x1 * w, x2 * w
			y1, y2 = y1 * h, y2 * h

			x_mean = (x1 + x2)/float(2)
			y_mean = (y1 + y2)/float(2)
			edge = max(y2 - y1, x2 - x1)* 1.5

			new_img = ut.plotTarget(img, (x_mean, y_mean, edge), ifSquareOnly = True)

			print "int(y_mean - edge/2.0), int(y_mean + edge/2.0), int(x_mean - edge/2.0), int(x_mean + edge/2.0)"
			print int(y_mean - edge/2.0), int(y_mean + edge/2.0), int(x_mean - edge/2.0), int(x_mean + edge/2.0)

			y_min = int(y_mean - edge/2.0) if int(y_mean - edge/2.0) >= 0 else 0
			y_max = int(y_mean + edge/2.0) if int(y_mean + edge/2.0) <= w else w
			x_min = int(x_mean - edge/2.0) if int(x_mean - edge/2.0) >= 0 else 0
			x_max = int(x_mean + edge/2.0) if int(x_mean + edge/2.0) <= h else h

			print "y_min, y_max, x_min, x_max: ", y_min, y_max, x_min, x_max
			# y_min = int(y_mean - edge/2.0)
			# y_max = int(y_mean + edge/2.0)
			# x_min = int(x_mean - edge/2.0)
			# x_max = int(x_mean + edge/2.0)
			print "img.shape: ", img.shape

			crop_img = img[ y_min : y_max,  x_min : x_max ] # Crop from x, y, w, h -> 100, 200, 300, 400
			# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
			print "crop_img.shape: ", crop_img.shape

			if debug:
				cv2.imshow("img", img)
				cv2.imshow("newImg", new_img)
				cv2.imshow("cropped", crop_img)

				cv2.waitKey(0)
			else:
				cv2.imwrite(output_dir + folder + "/" + img_name, crop_img)



# for index in range(1, len(lines)):
#     line = lines[index].replace("\r\n", "")
#     url = line
#     print url.split(" ")
#     try:
#         f = open( str(index) + ".jpg", 'wb')
#         img = urllib2.urlopen(url, timeout = 10).read()
#         if img != None:
#             f.write(img)
#         f.close()
#     except Exception as e:
#         print e
#     if index == 31:
#         break


# #
# line = lines[1]
# print line
# # url = line.split("\t")[2]
# print url
# f = open( "test.jpg", 'wb')
# img = urllib2.urlopen(url).read()
# f.write(img)
# f.close()
