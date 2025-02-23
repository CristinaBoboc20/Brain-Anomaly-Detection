import numpy as np 
import matplotlib.pyplot as mtp  
import pandas as pd 
from sklearn.preprocessing import StandardScaler    
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Incarcarea si preprocesarea datelor

train_labels= np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', delimiter=',',dtype=None,encoding=None )
print(train_labels)
train_labels=np.array(train_labels[1:])
train_images=[]
t_labels=[]
for elem in train_labels:
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/"+elem[0]+".png",  cv2.IMREAD_GRAYSCALE)
    t_labels.append(elem[1].astype(np.float64))
    train_images.append(file_img)
train_images=np.array(train_images)
t_labels=np.array(t_labels)
train_images=train_images.reshape(train_images.shape[0],train_images.shape[1], train_images.shape[2],1)
# Normalizare
train_images=train_images/255.0


validation_labels= np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', delimiter=',',dtype=None,encoding=None )

validation_labels=np.array(validation_labels[1:])
validation_images=[]
v_labels=[]
for elem in validation_labels:
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/"+elem[0]+".png",cv2.IMREAD_GRAYSCALE)
    v_labels.append((elem[1]).astype(np.float64))
    validation_images.append(file_img)
validation_images=np.array(validation_images)
v_labels=np.array(v_labels)
validation_images=validation_images.reshape(validation_images.shape[0],validation_images.shape[1], validation_images.shape[2],1)
validation_images=validation_images/255.0
print(validation_images)
test_images=[]
t_images=[]
for i in range(17001,22150):
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/0"+str(i)+".png",  cv2.IMREAD_GRAYSCALE)
    t_images.append(('0'+str(i)))
    test_images.append(file_img)
test_images=np.array(test_images)
test_images=test_images.reshape(test_images.shape[0],test_images.shape[1], test_images.shape[2],1)
test_images=test_images/255.0


# obiect de tip Sequential
model2=Sequential()
# adaugare straturi
model2.add(Conv2D(6, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))

model2.add(Conv2D(16, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))

model2.add(Conv2D(32, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))

model2.add(Conv2D(64, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))

model2.add(Conv2D(80, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))

model2.add(Conv2D(100, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))

model2.add(Conv2D(128, (5,5), padding='same', activation="relu",input_shape=(224, 224, 1)))
model2.add(MaxPooling2D((2, 2), strides=2))


model2.add(Flatten())
model2.add(Dense(units=120,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(units=80,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dense(units=10,activation='softmax'))
# se compilează modelul și se specifică funcția de pierdere, optimizatorul și metricile de performanță.
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#  funcție de callback care va salva cele mai bune rezultate ale modelului pe baza performanței acestuia pe setul de validare.
checkpoint = ModelCheckpoint('good.h5', monitor='valid_accuracy', save_best_only=True, mode='max', verbose=1)
model2.fit(train_images,t_labels, epochs=10, validation_data=(validation_images,v_labels), callbacks=[checkpoint])

# incarcarea versiunea modelului cu acuratetea cea mai buna pe validation_data
model3=load_model('good.h5')
model3.evaluate(validation_images,v_labels)
v_label10= model3.predict(validation_images)
# obținem clasele prezise pentru fiecare imagine din setul de validare
# valoarea maximă este găsită pe baza axei coloanelor din v_label10
v_label10 = np.argmax(v_label10, axis=1)
# calculare acuratețe, f1_score, matricea de confuzie, precizia si recall
acc= accuracy_score(v_labels, v_label10)
prec = precision_score(v_labels, v_label10)
recall = recall_score(v_labels, v_label10)
matrix = confusion_matrix(v_labels, v_label10)
f1 =  f1_score(v_labels, v_label10)


c= confusion_matrix(v_labels, v_label10)
tabel = ConfusionMatrixDisplay(confusion_matrix=c)
tabel.plot()


print(acc,prec,recall,f1)

test_labels = model3.predict(test_images)
test_labels = np.argmax(test_labels, axis=1)

# scriere fisier
with open('sample_submission.csv', 'w') as f:
    f.write('id,class\n')
    for i in range(len(test_labels)):
        f.write(f'{t_images[i]},{test_labels[i]}\n')
