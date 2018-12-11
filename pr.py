
    




	


#---------------------------------------------tkinter------------------------------

from tkinter import *
from tkinter import messagebox 
from tkinter import filedialog 
import os



from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import numpy as np

import c1

def takephoto1():
 c1.takephoto()
 return



def helloCallBack(argv):
 msg=messagebox.showinfo( "Result",argv)
 


def openme3(newArr):
 print('rahul')
 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 image_index = 7777 
 #print(y_train[image_index])
 plt.imshow(x_train[image_index], cmap='Greys')
 #plt.show()
 x_train.shape
 x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
 x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
 input_shape = (28, 28, 1)
 x_train = x_train.astype('float32')
 x_test = x_test.astype('float32')
 x_train /= 255
 x_test /= 255
 #print('x_train shape:', x_train.shape)
 #print('Number of images in x_train', x_train.shape[0])
 #print('Number of images in x_test', x_test.shape[0])
 model = Sequential()
 model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Flatten())
 model.add(Dense(128, activation=tf.nn.relu))
 model.add(Dropout(0.2))
 model.add(Dense(10,activation=tf.nn.softmax))
 model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
 fname="weights-test-CNN.hdf5"
 model.load_weights(fname)
 #model.evaluate(x_test, y_test)
 image_index = 4444
 plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
 img_rows=28
 img_cols=28
 pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
 #print(pred.argmax())
 #plt.show()
#  --------------my example----------------
 myarray = np.asarray(newArr)
 plt.imshow(myarray.reshape(28, 28),cmap='Greys')
 img_rows=28
 img_cols=28
 pred = model.predict(myarray.reshape(1, img_rows, img_cols, 1))
 print(pred.argmax())
 #plt.show()
 #w = Message(root, text=pred.argmax())
 #w.pack()
 helloCallBack(pred.argmax())
 #B = Button(top, text ="Hello", command = helloCallBack)
 #B.place(x=50,y=50)
 return








def openme2(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    argv=filedirectory
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))

    if width > height:
        nheight = int(round((20.0 / width * height), 0))  
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))  
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    #print(tva)
    return tva

def openme1(path):
    x=[openme2(path)]
    #print('here the length of x',len(x))
    #print(x[0])
    newArr=[[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
     for j in range(28):
        newArr[i][j]=x[0][k]
        k=k+1
    '''for i in range(28):
     for j in range(28):
        print(newArr[i][j])
        # print(' , ')
     print('\n')
    print('here the length of new',len(newArr))'''
    openme3(newArr)
    return






filename=""
filedirectory=""
def open_file():
 result = filedialog.askopenfile(initialdir="/", title="select file", filetypes=(("text files", ".txt"), ("all files", "*.*")))
 print(result)
 if result:
  #print(result.name)
  global filedirectory
  global filename
  filedirectory=result.name
  filename=os.path.basename(filedirectory)
  filepath=str(filedirectory)
 else:
  print('Please Select the file first')

root = Tk()

button = Button(root, text="open file", command=open_file)
button.pack()
button.place(x="100",y="50")
root.geometry("300x300")

#button1 = Button(root, text="checkfile", command=openme1)
button1 = Button(root, text='checkfile', command= lambda: openme1(filedirectory))
button1.pack()
button1.place(x="100",y="80")

button2 = Button(root, text='capture', command=takephoto1)
button2.pack()
button2.place(x="100",y="120")

root.geometry("400x400")

r = Message(root, text="Hello")
r.pack()
r.place(x="100",y="0")
r = Message(root, text="World")
r.pack()
r.place(x="140",y="0")
root.mainloop()


#-----------------------------------------tkinter ends-----------------------------




