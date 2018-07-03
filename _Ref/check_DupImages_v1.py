# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

def classify_gray_hist(image1,image2,size = (256,256)):
 image1 = cv2.resize(image1,size)
 image2 = cv2.resize(image2,size)
 hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0])
 hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0])
 plt.plot(range(256),hist1,'r')
 plt.plot(range(256),hist2,'b')
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

def classify_aHash(image1,image2):
 image1 = cv2.resize(image1,(8,8))
 image2 = cv2.resize(image2,(8,8))
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
 hash1 = getHash(gray1)
 hash2 = getHash(gray2)
 return Hamming_distance(hash1,hash2)

def classify_pHash(image1,image2):
 image1 = cv2.resize(image1,(32,32))
 image2 = cv2.resize(image2,(32,32))
 gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
 gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
 dct1 = cv2.dct(np.float32(gray1))
 dct2 = cv2.dct(np.float32(gray2))
 dct1_roi = dct1[0:8,0:8]
 dct2_roi = dct2[0:8,0:8]
 hash1 = getHash(dct1_roi)
 hash2 = getHash(dct2_roi)
 return Hamming_distance(hash1,hash2)

def getHash(image):
 avreage = np.mean(image)
 hash = []
 for i in range(image.shape[0]):
  for j in range(image.shape[1]):
   if image[i,j] > avreage:
    hash.append(1)
   else:
    hash.append(0)
 return hash

def Hamming_distance(hash1,hash2):
 num = 0
 for index in range(len(hash1)):
  if hash1[index] != hash2[index]:
   num += 1
 return num


def UnitTest(comp, thredhold = 0.9):
     #comp = []
     #comp.append('./tiff_1.tif')
     #comp.append('./tiff_2.tif')
     '''
     1.0
     1.0
     0
     0
     '''
     ret = ""
     img1 = cv2.imread(comp[0])
     #cv2.imshow('img1',img1)
     img2 = cv2.imread(comp[1])
     #cv2.imshow('img2',img2)
     degree = classify_gray_hist(img1,img2);print(degree)
     if( degree >= thredhold):
         ret = str(comp[0]) + "," + str(comp[1])
     degree = classify_hist_with_split(img1,img2); print(degree)
     degree = classify_aHash(img1,img2); print(degree)
     degree = classify_pHash(img1,img2); print(degree)
     #print(degree)
     return ret #cv2.waitKey(0)


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
            comp.clear();
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
                 print("[Ret]>", ret )
                 StringAddFile(ret, LOG_file)

                 comp.clear()
                 #if(".tif" in file.lower()):
                 comp.append(filepath)
                 isFirst = False


