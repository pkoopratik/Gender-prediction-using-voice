from django.shortcuts import render

import os
# Create your views here.
def index(request):
    return render(request,'index.html')

def male(request):
    return render(request,'male.html')
def female(request):
    return render(request,'female.html')        

def uploadfile(request):
    print("uploading file")
    dir = 'E:\\arti\\AI Project2\\AI\\aipro\\aipro'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    print(request.FILES['image'])
    f=request.FILES['image']
    from . models import user
    user=user(fl=f)
    user.save()
    import matplotlib.pyplot as plt
    plt.ioff()
    import pickle
    # load the model from disk
    import sklearn
    #print(os.listdir())
    print(sklearn.__version__)

    loaded_classifier = pickle.load(open('finalised_model.sav', 'rb'))
    print(sklearn.__version__)

    from live_extraction import extract_features
    #from record_audio import record

    path='E:\\AI\\aipro\\aipro\\'

    #record(path)
    extract_features(path)

    # Importing the libraries
    import pandas as pd
    import numpy as np

    #Importing the dataset
    original_dataset=pd.read_csv('features.csv')
    test_dataset = pd.read_csv('recorded_audio_features.csv')

    original_dataset=original_dataset.iloc[:,1:-1]
    test_dataset=test_dataset.iloc[:,1:]

    appended_dataset=original_dataset.append(test_dataset)

    X = appended_dataset.iloc[:,:].values

    print('Loaded Test Case')
    #Taking care of missing data
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=0, strategy='most_frequent')          
    imputer = imputer.fit(X[:,:])
    X[:,:] = imputer.transform(X[:,:])

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_test=np.array(X[-1])
    X_test=X_test.reshape(1,111)

    print('Predicting ...')
    # Predicting the Test set results
    y_pred = loaded_classifier.predict(X_test)

    if y_pred[0]==1:
        print('Male voice has been predicted')
        return render(request,'male.html')

    else:
        print('Female voice has been predicted')
        return render(request,'female.html')
    
   # return render(request,'index.html')



