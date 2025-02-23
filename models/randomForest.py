import numpy as np 
import matplotlib.pyplot as mtp  
import pandas as pd 
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler    
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Incarcarea si preprocesarea datelor
train_labels= np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', delimiter=',',dtype=None,encoding=None )

train_labels=np.array(train_labels[1:])
train_images=[]
t_labels=[]
for elem in train_labels:
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/"+elem[0]+".png",  cv2.IMREAD_GRAYSCALE)
    t_labels.append(elem[1].astype(np.float64))
    file_img=file_img.astype(np.float64)
    # imaginea va avea format (224,224)
    file_img=cv2.resize(file_img,(224,224))
    train_images.append(file_img)
train_images=np.array(train_images)
t_labels=np.array(t_labels)
train_images=train_images.reshape(len(train_images),-1)
# Normalizare
train_images=train_images/255.0

validation_labels= np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', delimiter=',',dtype=None,encoding=None )

validation_labels=np.array(validation_labels[1:])
validation_images=[]
v_labels=[]
for elem in validation_labels:
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/"+elem[0]+".png",cv2.IMREAD_GRAYSCALE)
    v_labels.append((elem[1]).astype(np.float64))
    file_img=file_img.astype(np.float64)
    file_img=cv2.resize(file_img,(224,224))
    validation_images.append(file_img)
validation_images=np.array(validation_images)
v_labels=np.array(v_labels)
validation_images=validation_images.reshape(len(validation_images),-1)
# Normalizare
validation_images=validation_images/255.0

test_images=[]
t_images=[]
for i in range(17001,22150):
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/0"+str(i)+".png",  cv2.IMREAD_GRAYSCALE)
    t_images.append(('0'+str(i)))
    file_img=file_img.astype(np.float64)
    file_img=cv2.resize(file_img,(224,224))
  
    test_images.append(file_img)
test_images=np.array(test_images)
test_images=test_images.reshape(len(test_images),-1)
# Normalizare
test_images=test_images/255.0

# Scalarea imaginilor
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
validation_images = scaler.transform(validation_images)
test_images = scaler.transform(test_images)
# aplicare randomforest

model = RandomForestClassifier(n_estimators=1000,max_depth=30)

model.fit(train_images, t_labels)
# predictie pe baza imaginilor din validation_images
v_label2=model.predict(validation_images)
# calculare acuratețe, f1_score, matricea de confuzie, precizia si recall
acc= accuracy_score(v_labels, v_label2)
prec = precision_score(v_labels, v_label2)
recall = recall_score(v_labels, v_label2)
matrix = confusion_matrix(v_labels, v_label2)
f1 =  f1_score(v_labels, v_label2)


c= confusion_matrix(v_labels, v_label2)
tabel = ConfusionMatrixDisplay(confusion_matrix=c)
tabel.plot()



test_labels = model.predict(test_images)

with open('sample_submission.csv', 'w') as f:
    f.write('id,class\n')
    for i in range(len(test_labels)):
        f.write(f'{t_images[i]},{test_labels[i]}\n')