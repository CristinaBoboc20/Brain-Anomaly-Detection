
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
import cv2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_labels= np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', delimiter=',',dtype=None,encoding=None )
train_labels=np.array(train_labels[1:])
train_images=[]
t_labels=[]
for elem in train_labels:
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/"+elem[0]+".png",  cv2.IMREAD_GRAYSCALE)
    t_labels.append(elem[1].astype(int))
    file_img=cv2.resize(file_img,(224,224))
    file_img=file_img/255.0
    file_img=file_img.flatten()
    train_images.append(file_img)
train_images=np.array(train_images)
t_labels=np.array(t_labels)
print(train_images)

validation_labels= np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', delimiter=',',dtype=None,encoding=None )

print(validation_labels)
validation_labels=np.array(validation_labels[1:])
validation_images=[]
v_labels=[]
for elem in validation_labels:
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/"+elem[0]+".png",cv2.IMREAD_GRAYSCALE)
    v_labels.append((elem[1]).astype(int))
#     file_img=file_img.astype(np.float64)
    file_img=cv2.resize(file_img,(224,224))
    file_img=file_img/255.0
    file_img=file_img.flatten()
    validation_images.append(file_img)
validation_images=np.array(validation_images)
v_labels=np.array(v_labels)
print(validation_images)

test_images=[]
t_images=[]
for i in range(17001,22150):
    file_img=cv2.imread("/kaggle/input/unibuc-brain-ad/data/data/0"+str(i)+".png",  cv2.IMREAD_GRAYSCALE)
    t_images.append(('0'+str(i)))
#     file_img=file_img.astype(np.float64)
    file_img=cv2.resize(file_img,(224,224))
    file_img=file_img/255.0
    file_img=file_img.flatten()
    test_images.append(file_img)
test_images=np.array(test_images)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(train_images, t_labels)



v_label2=model.predict(validation_images)

acc= accuracy_score(v_labels, v_label2)
prec = precision_score(v_labels, v_label2)
recall = recall_score(v_labels, v_label2)
matrix = confusion_matrix(v_labels, v_label2)
f1 =  f1_score(v_labels, v_label2)


c= confusion_matrix(v_labels, v_label2)
tabel = ConfusionMatrixDisplay(confusion_matrix=c)
tabel.plot()


print(acc,prec,recall,f1)


# predictii pe baza iamginilor din test
test_labels = model.predict(test_images)
# scriere fisier
with open('sample_submission.csv', 'w') as f:
    f.write('id,class\n')
    for i in range(len(test_labels)):
        f.write(f'{t_images[i]},{test_labels[i]}\n')