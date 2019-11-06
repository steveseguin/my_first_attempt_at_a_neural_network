import cv2
import numpy as np
from sys import argv
import PIL.Image as pil
from urllib2 import urlopen
from  cStringIO import StringIO
import cv2.cv as cv
import os
import time

thought = np.zeros((28*28)) 
weightsInput = np.zeros((28*28,28*28))
weightsOut = np.zeros((28*28,10)) ## 
output = np.zeros((10))
answer = np.zeros((10))

noise1 = np.random.normal(1,1,28*28*10)
noise1 = np.reshape(noise1, (28*28,10))
weightsOut = noise1
weightsOut = np.clip(weightsOut, 0, 1)

noise2 = np.random.normal(1,1,28*28*28*28)
noise2 = np.reshape(noise2, (28*28,28*28))
weightsInput = noise2
weightsInput = np.clip(weightsInput, 0, 1)
print "starting..."
img2  = [0,1,2,3,4,5,6,7,8,9]
for root, dirs, files in os.walk("C:\\convolution\\"):
				for file in files:
						if ((file.endswith(".png")) | (file.endswith(".jpg"))):
							filename = os.path.join(root, file)
							im = pil.open(filename) #use PIL to download and open an image from the WWW
							im = np.asarray(im)
							label = int(file.split(".")[0])
							img2[label] =im
print "processing done"
bestResponse = -1
for zzz in range(0,10):
	for x in range(0,70):
		for y in range(0,70):
			response = 0
			for l in range(0,10):
				print l,
				repeat = 1
				
				answer = np.zeros((10))
				answer[l] = 1.0
				
				#img = img2[l]
				num = img2[l][x*28:(x+1)*28,y*28:(y+1)*28]  ## since the dataset is one giant image, break it down to the sub-image
				visible_input = np.reshape(num, (28*28))  ## linearize the image.  cause my neurons be like that
				thought[:] = 1.0  ## reset thoughts for each new image
				output[:] = 1.0
				
				rr = 0
				for j in range(0,28*28): ## OUTPUT 
					for i in range(0,28*28):  ## INPUT
						if ((weightsInput[i,j] + visible_input[i])!=0):
							thought[j] *= (1.0 - (2.0*weightsInput[i,j]*visible_input[i])/(weightsInput[i,j] + visible_input[i]))
					thought[j] = 1.0 - thought[j]
				
				for j in range(0,10):   ## OUTPUT, for each output, this can be done in parallel
					for i in range(0,28*28):  ## INPUT, can be done in serial or mostly-parallel 
						if ((weightsOut[i,j] + thought[i])!=0):
							output[j] += (1.0 - (2.0*weightsOut[i,j]*thought[i])/(weightsOut[i,j] + thought[i]))
					output[j] = 1.0 - output[j]/(28*28) ## final step for each output neuron
					rr += pow(output[j]-answer[j],2)
				response+=pow(rr,2)
			if bestResponse == -1:
				print "initial check"
				bestResponse = response
				bestWeightsOut = weightsOut
				bestweightsInput = weightsInput
				noise1 = np.random.normal(0,response,28*28*10)
				noise1 = np.reshape(noise1, (28*28,10))
				noise2 = np.random.normal(0,response,28*28*28*28)
				noise2 = np.reshape(noise2, (28*28,28*28))
			elif response <= bestResponse:
				bestResponse = response
				bestWeightsOut = weightsOut
				bestweightsInput = weightsInput
				print ""
				print "reponse",response
			else:
				response = bestResponse
				weightsOut = bestWeightsOut
				weightsInput = bestweightsInput
				print ".",
				noise1 = np.random.normal(0,response,28*28*10)
				noise1 = np.reshape(noise1, (28*28,10))
				noise2 = np.random.normal(0,response,28*28*28*28)
				noise2 = np.reshape(noise2, (28*28,28*28))
			
			#response /= 10
			
			
			weightsOut += noise1
			weightsOut = np.clip(weightsOut, 0, 1)
			weightsInput += noise2
			weightsInput = np.clip(weightsInput, 0, 1)
		print ""
		print "output:",output
			#print weightsOut
										
									