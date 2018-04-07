# USAGE
# run in terminal:
# python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel


#uses code from Adrian Rosebrock at PyImageSearch : https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/
# import the necessary packages
#import weakref
import cv2
import numpy as np
import argparse
import sys
from Tkinter import *
import tkFileDialog
import requests
from urllib2 import urlopen
from PIL import Image, ImageTk
from selenium import webdriver
import io
import base64

#def main():

root = Tk()
root.withdraw() # Close the root window

path = tkFileDialog.askopenfilename()
root.update()
root.destroy()
root.quit()
#print in_path


	#print(filename)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
	#help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
#image = cv2.imread(args["image"])

#image = cv2.imread("/Users/reidyesson/Documents/object-detection-deep-learning/images/standingMan2.jpg")
image = cv2.imread(path)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
crop_img = []

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		crop_img = image[startY:endY, startX:endX]

def returnAllImages(img):
		height, width = img.shape[:2]
		shirtTop = height // 6
		shirtBottom = height // 2
		shirt = img[shirtTop:shirtBottom]
		shoesTop = (9 * height) // 10
		shoes = img[shoesTop:height, 0:width]
		newshirt = Image.fromarray(shirt, 'RGB')
		newshoes = Image.fromarray(shoes, 'RGB')
		newshirt.save('shirt.gif')
		newshoes.save('shoes.gif')
		return ('shirt.gif', 'shoes.gif')

def getInfo(imageFile, length=8):
	filePath = imageFile
	searchUrl = 'http://www.google.com/searchbyimage/upload'
	multipart = {'encoded_image': (filePath, open(filePath, 'rb')),
		'image_content': ''}
	#multipart = {'encoded_image': (imageFile),
		#'image_content': ''}


	response = requests.post(searchUrl, files=multipart, allow_redirects=False)
	fetchUrl = response.headers['Location']

	browser = webdriver.Chrome(executable_path = '/usr/local/bin/chromedriver')
	browser.get(fetchUrl) #navigate to the page
	button = browser.find_element_by_css_selector(
		"div.hdtb-mitem:nth-child(4) > a:nth-child(1)")
	button.click()

	r = requests.get(browser.current_url)

	#browser.quit()
    
	itemList = []
	priceList = []

	for i in range(len(r.text)):
		if r.text[i] == "$" and r.text[i-3:i] == "<b>":
			j = i
			s = ''
			while r.text[j] != "<":
				s += r.text[j]
				j += 1
			j = i
			link = ''
			while r.text[j-2:j] != "/a":
				j -= 1
			j -= 5
			while r.text[j:j+4] != 'http':
				j -= 1
				link = r.text[j] + link
			if len(priceList) < length:
				priceList.append(s)
				itemList.append(link)
	#itemList = [imageFile] + [itemList]
	return [imageFile, itemList, priceList]


shirtAndShoes = returnAllImages(crop_img)
shirtList = getInfo(shirtAndShoes[0])
shoesList = getInfo(shirtAndShoes[1])


class ImageUrl(object):
	def __init__(self, url):
		picture = urlopen(url).read()
		imageFile = io.BytesIO(picture)
		pilImg = Image.open(imageFile)
		self.image = ImageTk.PhotoImage(pilImg)
		self.height = self.image.height()
		self.width = self.image.width()

class ImageFle(object):
	def __init__(self, file):
		self.file = file
		self.image = PhotoImage(file=self.file)
		self.height = self.image.height()
		self.width = self.image.width()

def init(data):
	data.oShirt = ImageFle(shirtList[0])
	data.shirt1 = ImageUrl(shirtList[1][0])
	data.shirt2 = ImageUrl(shirtList[1][1])
	data.shirt3 = ImageUrl(shirtList[1][2])
	data.shirt4 = ImageUrl(shirtList[1][3])
	data.shirt5 = ImageUrl(shirtList[1][4])
	data.shirt6 = ImageUrl(shirtList[1][5])
	data.shirt7 = ImageUrl(shirtList[1][6])
	data.shirt8 = ImageUrl(shirtList[1][7])
	data.oShoes = ImageFle(shoesList[0])
	data.shoes1 = ImageUrl(shoesList[1][0])
	data.shoes2 = ImageUrl(shoesList[1][1])
	data.shoes3 = ImageUrl(shoesList[1][2])
	data.shoes4 = ImageUrl(shoesList[1][3])
	data.shoes5 = ImageUrl(shoesList[1][4])
	data.shoes6 = ImageUrl(shoesList[1][5])
	data.shoes7 = ImageUrl(shoesList[1][6])
	data.shoes8 = ImageUrl(shoesList[1][7])
	data.originalList = [data.oShirt, data.oShoes]
	data.matchesList = [data.shirt1,data.shirt2,data.shirt3,data.shirt4,data.shirt5,data.shirt6,data.shirt7,
						data.shirt8,data.shoes1,data.shoes2,data.shoes3,data.shoes4,data.shoes5,data.shoes6,
						data.shoes7,data.shoes8]
	data.priceLists = [shirtList[2][0],shirtList[2][1],shirtList[2][2],shirtList[2][3],shirtList[2][4],shirtList[2][5],shirtList[2][6],shirtList[2][7],
						shoesList[2][0],shoesList[2][1],shoesList[2][2],shoesList[2][3],shoesList[2][4],shoesList[2][5],shoesList[2][6],shoesList[2][7]]
	data.startScreen = True
	data.shirtScreen = False
	data.shirtAdvanced = False
	data.shoesScreen = False
	data.shoesAdvanced = False

def mousePressed(event, data):
    # use event.x and event.y
    pass

def keyPressed(event, data):
    if data.startScreen == True:
        if event.keysym == "Right":
            data.startScreen = False
            data.shirtScreen = True
    elif data.shirtScreen == True:
        if event.keysym == "Right":
            data.shirtScreen = False
            data.shoesScreen = True
        elif event.keysym == "Down":
        	data.shirtScreen = False
        	data.shirtAdvanced = True
    elif data.shirtAdvanced == True:
    	if event.keysym == "Left":
    		data.shirtAdvanced = False
    		data.shirtScreen = True
    	elif event.keysym == "Right":
    		data.shirtAdvanced = False
    		data.shoesScreen = True
    elif data.shoesScreen == True:
        if event.char == "r":
            init(data)
        elif event.keysym == "Left":
            data.shoesScreen = False
            data.shirtScreen = True
        elif event.keysym == "Down":
        	data.shoesScreen = False
        	data.shoesAdvanced = True
    elif data.shoesAdvanced == True:
    	if event.keysym == "Left":
    		data.shoesAdvanced = False
    		data.shoesScreen = True
    	elif event.char == "r":
            init(data)

def drawStartScreen(canvas, data):
    title = "~Clothzam~"
    description = "See it, search it, wear it!"
    fontSizeT = 140
    fontSizeD = 60
    yPropStart = 0.4
    desProp = 0.6
    canvas.create_rectangle(0,0, data.width, data.height,
                            fill="peach puff",width= 0)
    canvas.create_text(data.width//2,data.height*yPropStart,text=title,
                        fill="black",font="msserif %d" %fontSizeT)
    canvas.create_text(data.width//2,data.height*desProp,text=description,
                        fill="black",font="msserif %d" %fontSizeD)
    
    #data.button = Button(canvas, text = "Click to get Started!", command = shirtButton(data))
    #data.button.place(relx=0.5, rely=0.5, anchor=CENTER)

def shirtButton(data):
	data.startScreen = False
	data.shirtScreen = True

def drawShirtScreen(canvas, data):
    title = "Shirt"
    text1 = "Actual Picture:"
    text2 = "3 Best Matches:"
    margin = 10
    textPropA = 0.15
    textPropB = 0.08
    fontSizeA = 40
    fontSizeB = 70
    canvas.create_rectangle(0,0, data.width, data.height,
                            fill="peach puff",width= 0)
    canvas.create_text(data.width//2,data.height*textPropB,text=title,
                        fill="black",font="msserif %d bold underline" %fontSizeB)
    canvas.create_text(margin,data.height*textPropA,text=text1,
                        fill="black",font="msserif %d" %fontSizeA, anchor=NW)
    canvas.create_text(data.width - margin,data.height*textPropA,text=text2,
                        fill="black",font="msserif %d" %fontSizeA, anchor=NE)
    canvas.create_image(data.width//4, data.height//2, image=data.originalList[0].image)
    for i in range(3):
    	fontSize = 20
    	price = "Price: %s" %(data.priceLists[i])
    	yImage = (0.27*data.height)*(i+1)
    	xCoord = (data.width//4)*3
    	yText = yImage + data.matchesList[i].height
    	canvas.create_image(xCoord, yImage,image=data.matchesList[i].image)
    	canvas.create_text(xCoord, yText, text=price, fill="black",font="msserif %d" %fontSize)

def drawShirtAdvanced(canvas, data):
	title = "Similar Shirts Results:"
	textPropA = 0.08
	fontSizeA = 60
	canvas.create_rectangle(0,0, data.width, data.height,
		fill="peach puff",width= 0)
	canvas.create_text(data.width//2,data.height*textPropA,text=title,
		fill="black",font="msserif %d bold underline" %fontSizeA)
	for i in range(3,8):
		fontSize = 20
		xCoord = (data.width//6)*(i+1-3)
		yText = (data.height//4)*3
		price = "Price: %s" %(data.priceLists[i])
		canvas.create_image(xCoord, data.height//2, image=data.matchesList[i].image)
		canvas.create_text(xCoord, yText, text=price, fill="black",font="msserif %d" %fontSize)

def drawShoesScreen(canvas, data):
    title = "Shoes"
    text1 = "Actual Picture:"
    text2 = "3 Best Matches:"
    margin = 10
    textPropA = 0.15
    textPropB = 0.08
    fontSizeA = 40
    fontSizeB = 70
    canvas.create_rectangle(0,0, data.width, data.height,
                            fill="peach puff",width= 0)
    canvas.create_text(data.width//2,data.height*textPropB,text=title,
                        fill="black",font="msserif %d bold underline" %fontSizeB)
    canvas.create_text(margin,data.height*textPropA,text=text1,
                        fill="black",font="msserif %d" %fontSizeA, anchor=NW)
    canvas.create_text(data.width - margin,data.height*textPropA,text=text2,
                        fill="black",font="msserif %d" %fontSizeA, anchor=NE)
    canvas.create_image(data.width//4, data.height//2, image=data.originalList[1].image)
    for i in range(8,11):
    	fontSize = 20
    	price = "Price: %s" %(data.priceLists[i])
    	yImage = (0.27*data.height)*(i+1-8)
    	xCoord = (data.width//4)*3
    	yText = yImage + data.matchesList[i].height
    	canvas.create_image(xCoord, yImage,image=data.matchesList[i].image)
    	canvas.create_text(xCoord, yText, text=price, fill="black",font="msserif %d" %fontSize)

def drawShoesAdvanced(canvas, data):
	title = "Similar Shoes Results:"
	textPropA = 0.08
	fontSizeA = 60
	canvas.create_rectangle(0,0, data.width, data.height,
		fill="peach puff",width= 0)
	canvas.create_text(data.width//2,data.height*textPropA,text=title,
		fill="black",font="msserif %d bold underline" %fontSizeA)
	for i in range(11,16):
		fontSize = 20
		xCoord = (data.width//6)*(i+1-11)
		yText = (data.height//4)*3
		price = "Price: %s" %(data.priceLists[i])
		canvas.create_image(xCoord, data.height//2, image=data.matchesList[i].image)
		canvas.create_text(xCoord, yText, text=price, fill="black",font="msserif %d" %fontSize)

def redrawAll(canvas, data):
    if data.startScreen == True:
        drawStartScreen(canvas, data)
    elif data.shirtScreen == True:
        drawShirtScreen(canvas, data)
    elif data.shirtAdvanced == True:
        drawShirtAdvanced(canvas, data)
    elif data.shoesScreen == True:
        drawShoesScreen(canvas, data)
    elif data.shoesAdvanced == True:
        drawShoesAdvanced(canvas, data)

####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    root = Tk()
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    redrawAll(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(800, 800)




