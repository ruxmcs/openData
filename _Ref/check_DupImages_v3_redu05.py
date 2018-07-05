# -*- coding: utf-8 -*-

import cv2
#import numpy as np
#from matplotlib import pyplot as plt

def classify_gray_hist(image1,image2,size = (256,256)):
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
 #plt.plot(range(256),hist1,'r')
 #plt.plot(range(256),hist2,'b')
 #plt.show()
 degree = 0
 for i in range(len(hist1)):
  if hist1[i] != hist2[i]:
   degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
  else:
   degree = degree + 1
 degree = degree/len(hist1)
 return degree

def calculate(image1,image2):
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
 degree = 0
 for i in range(len(hist1)):
  if hist1[i] != hist2[i]:
   degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
  else:
   degree = degree + 1
 degree = degree/len(hist1)
 return degree

def classify_hist_with_split(image1,image2,size = (256,256)):
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 sub_image1 = cv2.split(image1)
 sub_image2 = cv2.split(image2)
 sub_data = 0
 for im1,im2 in zip(sub_image1,sub_image2):
  sub_data += calculate(im1,im2)
 sub_data = sub_data/3
 return sub_data


def Hamming_distance(hash1,hash2):
 num = 0
 for index in range(len(hash1)):
  if hash1[index] != hash2[index]:
   num += 1
 return num


def UnitTest(comp, thredhold = 0.5):
     ret = ""
     img1 = cv2.imread(comp[0])
     img2 = cv2.imread(comp[1])
     degree = classify_gray_hist(img1,img2);print(degree)
     if( degree >= thredhold):
         ret = str(comp[0]) + "," + str(comp[1])
     degree = classify_hist_with_split(img1,img2); print(degree)
     return ret


def StringAddFile(urlde, filepath):
    file = open(filepath, "a")
    file.write(urlde)
    file.close()
import os
if __name__ == '__main__':

     print("Unit-Test-1: same")
     comp = []
     comp.append('./tiff_1.tif')
     comp.append('./tiff_2.tif')
     UnitTest(comp)
     print("Unit-Test-1: diff")
     comp.clear()
     comp.append('./tiff_1.tif')
     comp.append('./tiff_a.tif')
     UnitTest(comp)
     print("Done with Unit-Test")

     BASE_dir = "./"
     LOG_file = "./Log_comp.txt"
     isFirst = True
     for file in os.listdir(BASE_dir):
         if(isFirst):
            filepath = BASE_dir +file
            comp.clear(); print("[Begin]<===")
            print("[0]>", filepath )
            if(".tif" in file.lower()):
                comp.append(filepath)
                isFirst = False
         else:
             filepath = BASE_dir +file
             if(".tif" in file.lower()):
                 print("[1]>", filepath)
                 comp.append( filepath )
                 ret = UnitTest(comp)
                 if(len(ret)>1):
                     print("[Result]========>", ret )
                     StringAddFile(ret, LOG_file)
                 else:
                     print("")

                 comp.clear()
                 print("[Begin]<===")
                 #if(".tif" in file.lower()):
                 comp.append(filepath)
                 isFirst = False


