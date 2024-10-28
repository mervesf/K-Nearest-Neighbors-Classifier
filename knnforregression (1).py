# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 07:29:16 2023

@author: user
"""
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt


def find_indices(array,k):
    new_data=pd.DataFrame(array).sort_values(by=0,ascending=True)
    return new_data.iloc[:k,:].index

def find_target(indices,y_labels):
    weights=[i for i in np.arange(0.02,0.001,-0.001)]
    result=list(y_labels.iloc[indices].values)
    return result 

def build_knn_model(x_train,y_train):
    x_samples=x_train
    y_labels=y_train
    return x_samples,y_labels


def predict_knn(x_samples,y_labels,x_test,k):
    distance_list=euclidean(x_samples,x_test)
    y_head=[]
    for i in distance_list:
        indices=find_indices(i,k)
        y_head.append(find_target(indices,y_labels))
        
        
    return y_head
        
def euclidean(x_data,x_data2):
        distance=[np.sum(np.power(i.reshape(1,-1)-np.array(x_data),2),axis=1) for i in np.array(x_data2)]
        return distance

def knn_classification(k,x_train,y_train,x_test,y_test):
    x_train=x_train.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    x_test=x_test.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    x_samples,y_labels=build_knn_model(x_train,y_train)
    y_head=predict_knn(x_samples,y_labels,x_test,k)
    print(accuracy_score(y_test, y_head))
    
knn_classification(3,x_train,y_train,x_test,y_test)