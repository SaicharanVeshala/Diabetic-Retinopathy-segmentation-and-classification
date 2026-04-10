from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt

main = tkinter.Tk()#blank page 
main.title("retiopathy segmentation and classification")
main.geometry("1300x1200")

global filename
global classifier
global labels
global X_train, Y_train

def readLabels(filename):
    global labels
    labels = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)


def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")#select the folder
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    readLabels(filename)
    text.insert(END,"Types of Retinopathy Diabetes diseases found in dataset is : "+str(labels)+"\n")

def preprocessDataset():
    global X_train, Y_train
    text.delete('1.0', END)
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    text.insert(END,"CNN is training on total Retinopathy Diabetes disease images : "+str(len(X_train))+"\n")
    test = X_train[0]
    cv2.imshow("Sample Loaded Image",cv2.resize(test,(300,300)))
    cv2.waitKey(0)



def trainCNN():
    global classifier
    global X_train, Y_train
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[49] * 100
        text.insert(END,"Retinopathy Diabetes Training Model Prediction Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 5, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[49] * 100
        text.insert(END,"Retinopathy Diabetes Training Model Prediction Accuracy = "+str(accuracy))

    

def predictDisease():
    filename = filedialog.askopenfilename(initialdir="testImages")#asks the image to select
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)#used to predict the disease on categories
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Diabetes Disease Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Diabetes Disease Predicted as : '+labels[predict], img)
    cv2.waitKey(0)
    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.plot(accuracy, 'ro-', color = 'orange')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Diabetes Retinopathy CNN Training Accuracy & Loss Graph')
    plt.show()



def Exit():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='retiopathy segmentation and classification',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Diabetes Retinopathy Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

preprocess = Button(main, text="Preprocess Images", command=preprocessDataset)
preprocess.place(x=50,y=200)
preprocess.config(font=font1)  

trainButton = Button(main, text="Train Diabetes Images Using CNN", command=trainCNN)
trainButton.place(x=50,y=250)
trainButton.config(font=font1)

testButton = Button(main, text="Upload Test Image & Predict Disease", command=predictDisease)
testButton.place(x=50,y=300)
testButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)


ExitButton = Button(main, text="Close GUI", command=Exit)
ExitButton.place(x=50,y=400)
ExitButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='grey')
main.mainloop()
