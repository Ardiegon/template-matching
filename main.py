import os

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
from numpy.lib.type_check import imag
from copy import deepcopy


def infoImage(img, img_title="image"):
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")

def loadImage(name):
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    return img

def cvImshow(img, img_title="image"):
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img * 128
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(-1)

def cvHishow(hist, hist_title = "histogram"):
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()
    plt.clf()

def drawRect(img, start, end):
    return cv2.rectangle(img, start, end, (0,0,255), 2)

def subHist(imgA, imgB):
    histA = cv2.calcHist([imgA], [0], None, [256], [0, 256]).flatten()
    histB = cv2.calcHist([imgB], [0], None, [256], [0, 256]).flatten()
    normalizatorA = 100/sum(histA)
    normalizatorB = 100/sum(histB)
    mistake = 0
    for a, b in zip(histA, histB):
        mistake += abs((a*normalizatorA)**2-(b*normalizatorB)**2)
    return mistake

def scaleShapes(template_shape, max_shape, accuracy_step= 0.5, min_shape_perc = 0.5):
    sh = template_shape
    shapes = []
    min_shape = (int(sh[0]*min_shape_perc), int(sh[1]*min_shape_perc))
    minratio = max(min_shape[0]/sh[0], min_shape[1]/sh[1])
    maxratio = min(max_shape[0]/sh[0], max_shape[1]/sh[1])
    for r in np.arange(minratio, maxratio, accuracy_step):
        shapes.append((int(round(sh[0]*r)), int(round(sh[1]*r))))
    return shapes

def matchTemplate(template, image, filter_accuracy_step = 0.4, sample_pixel_step = 10, cv2color = cv2.COLOR_BGR2GRAY):
    shapes = scaleShapes(template.shape, image.shape, accuracy_step = filter_accuracy_step)
    print(shapes)
    samples = []
    for shid, shape in enumerate(shapes):
        wsteps = math.ceil((image.shape[1]-shape[1])/sample_pixel_step)
        hsteps = math.ceil((image.shape[0]-shape[0])/sample_pixel_step)
        for h in range(0, hsteps):
            if (h)%10 == 0:
                print(f"Shape: {shid+1} line: {h+1}/{hsteps}")
            for w in range(0, wsteps):
                ws = w*sample_pixel_step
                we = w*sample_pixel_step+shape[1]
                hs = h*sample_pixel_step
                he = h*sample_pixel_step+shape[0]
                samples.append([image[hs:he, ws:we], (hs,he,ws,we), None])
    samples = np.array(samples)
    for said, sample in enumerate(samples):
        if (said)%5000 ==0:
            print(f"Samples done: {said}/{len(samples)}")
        sample[2] = subHist(sample[0], template)
    best_sample_id = list(samples[:,2]).index(min(samples[:,2]))
    return samples[best_sample_id]

if __name__ == "__main__":
    imgA = loadImage("./images/pikachu.png")
    imgB = loadImage("./images/achu.png")
    imgC = loadImage("./images/ika.png")
    imgD = loadImage("./images/accantus.png")
    imgE = loadImage("./images/tus.png")
    imgF = loadImage("./images/acc2.png")
    imgG = loadImage("./images/photo.png")
    imgH = loadImage("./images/oto.png")

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    grayC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
    grayD = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)
    grayE = cv2.cvtColor(imgE, cv2.COLOR_BGR2GRAY)
    grayF = cv2.cvtColor(imgF, cv2.COLOR_BGR2GRAY)
    grayG = cv2.cvtColor(imgG, cv2.COLOR_BGR2GRAY)
    grayH = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)


    found = matchTemplate(grayH, grayG)
    imgfound = drawRect(imgG, (found[1][2],found[1][0]), (found[1][3],found[1][1]))

    cvImshow(imgfound)