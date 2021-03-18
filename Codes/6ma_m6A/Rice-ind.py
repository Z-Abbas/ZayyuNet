#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:14:55 2020

@author: zeeshan
"""



#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";

#import matplotlib as mpl
#mpl.use('Agg')

from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, Input,MaxPooling1D,Flatten,LeakyReLU,Activation,concatenate,Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
#from group_norm import GroupNormalization
import random
import pandas as pd 
import numpy as np
from keras import regularizers
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os, sys, copy, getopt, re, argparse
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import keras

np.random.seed(seed=21)
tf.__version__
keras.__version__

from keras import losses
import pickle
from scipy import interp

#tf.compat.v1.enable_eager_execution()



def analyze(temp, OutputDir):

    # temp = None
    # with open(dataFile, 'rb') as file:
    #        temp = pickle.load(file)

    trainning_result, validation_result, testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [trainning_result, validation_result, testing_result]:


        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;

        file.write(title +  'results\n')


        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:

            total = []

            for val in x:
                total.append(val[j])

            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n')

        file.write('\n\n______________________________\n')
    file.close();

    index = 0

    for x in [trainning_result, validation_result, testing_result]:

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0

        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        plt.savefig( OutputDir + '/' + title +'ROC.png')
        plt.close('all');

        index += 1;

##############################   Scheduler ########################
def scheduler(epochs, lr):
  if epochs < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

####################################################################
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    
    return out

def calculate(sequence):

    X = []
    dictNum = {'A' : 0, 'T' : 0, 'C' : 0, 'G' : 0};

    for i in range(len(sequence)):

        if sequence[i] in dictNum.keys():
            dictNum[sequence[i]] += 1;
            X.append(dictNum[sequence[i]] / float(i + 1));

    return np.array(X)

def dataProcessing(path):

    data = pd.read_csv(path);
    alphabet = np.array(['A', 'G', 'T', 'C','N'])
    X = [];
    for line in data['data']:

        line = list(line.strip('\n'));
        #scoreSequence = calculate2(line);
        
        seq = np.array(line, dtype = '|U1').reshape(-1,1);
        seq_data = []

        for i in range(len(seq)):
            if seq[i] == 'A':
                seq_data.append([1,0,0,0])
            if seq[i] == 'T':
                seq_data.append([0,1,0,0])
            if seq[i] == 'C':
                seq_data.append([0,0,1,0])
            if seq[i] == 'G':
                seq_data.append([0,0,0,1])
            if seq[i] == 'N':
                seq_data.append([0,0,0,0])
                
        X.append(np.array(seq_data));
        
    X = np.array(X);
    y = np.array(data['label'], dtype = np.int32);
 
    return X, y; #(n, 34, 4), (n,)

def prepareData(PositiveCSV, NegativeCSV):

    Positive_X, Positive_y = dataProcessing(PositiveCSV);
    Negitive_X, Negitive_y = dataProcessing(NegativeCSV);

    return Positive_X, Positive_y, Negitive_X, Negitive_y

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;

#*******************Arch 3***********************#
#************************************************#
def spinal_cnn():
    input_shape = (41,4)
    inputs = Input(shape = input_shape)
    
    
    conv0 = Conv1D(filters=64, kernel_size=5,strides=1)(inputs)
    normLayer0 = BatchNormalization()(conv0);
    #pool0 = MaxPooling1D(pool_size = 2)(normLayer0)
    act0 = Activation(activation='relu')(normLayer0)
    
    
    
    conv2 = Conv1D(filters=128, kernel_size=5,strides=1)(act0)
    normLayer2 = BatchNormalization()(conv2);
    pool2 = MaxPooling1D(pool_size = 2)(normLayer2)
    dropoutLayer1 = Dropout(0.30)(pool2)
    act2 = Activation(activation='relu')(dropoutLayer1)
    
    x = Flatten()(act2)
    
    
    conv4 = Conv1D(filters=128, kernel_size=5,strides=1)(act0)
    normLayer4 = BatchNormalization()(conv4);
    pool4 = MaxPooling1D(pool_size = 2)(normLayer4)
    dropoutLayer2 = Dropout(0.30)(pool4)
    act4 = Activation(activation='relu')(dropoutLayer2)
    
    
    a = Flatten()(act4)
    comb = concatenate([x, a], axis=1)
    
    a1 = keras.layers.Lambda(lambda comb: comb[:,0:2048], output_shape=(2048,))(comb)
    a2 = keras.layers.Lambda(lambda comb: comb[:,2048:], output_shape=(2048,))(comb)
    
    #x1 = x[:, 0:360]
    
    a1 = Dense(8, activation='relu')(a1) # Number of nodes in hidden layer
    
    a2 = concatenate([a2, a1])
    a2 = Dense(8, activation='relu')(a2)
    
    a3 = concatenate([a1, a2])
    a3 = Dense(8, activation='relu')(a3)
    
    a4 = concatenate([a2, a3])
    a4 = Dense(8, activation='relu')(a4)
    
    a5 = concatenate([a1, a4])
    a5 = Dense(8, activation='relu')(a3)
    
    a6 = concatenate([a2, a5])
    a6 = Dense(8, activation='relu')(a6)
    
    a = concatenate([a1, a2], axis=1)
    a = concatenate([a, a3], axis=1)
    a = concatenate([a, a4], axis=1)
    a = concatenate([a, a5], axis=1)
    a = concatenate([a, a6], axis=1)
    
    #xa = concatenate([a, x], axis=1)
    
    
    output = Dense(1, activation= 'sigmoid')(a)
    
    
    
    model = Model(inputs = inputs, outputs = output)
    opt=SGD(learning_rate=0.001, momentum = 0.95)
    model.compile(loss='binary_crossentropy', optimizer= opt, metrics=[binary_accuracy]);
    
    return model



def calculateScore(X, y, model, folds):
    
    score = model.evaluate(X,y)
    pred_y = model.predict(X)

    accuracy = score[1];

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN+FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    plt.show() 

    lossValue = losses.binary_crossentropy(y_true, y_pred)#.eval()

    return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea, 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'lossValue' : lossValue}

def funciton(PositiveCSV, NegativeCSV, OutputDir, folds):

    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(PositiveCSV, NegativeCSV)
    
    random.shuffle(Positive_X);
    random.shuffle(Negitive_X);

    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);

    trainning_result = []
    validation_result = []
    testing_result = []
    
    for test_index in range(folds):

        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index],Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index],Negative_y_Slices[validation_index]))

        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))

                
                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))
        print(np.shape(tempX),np.shape(train_X))
        test_X, test_y = shuffleData(test_X,test_y);
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y);
        
        print(np.shape(train_X), np.shape(valid_X), np.shape(test_X))
        
        np.save('/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rosaceae/chunk_folds/'+str(test_index)+'_'+'x_test',test_X)
        np.save('/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rosaceae/chunk_folds/'+str(test_index)+'_'+'y_test',test_y)
        np.save('/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rosaceae/chunk_folds/'+str(test_index)+'_'+'valid_X',valid_X)
        np.save('/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rosaceae/chunk_folds/'+str(test_index)+'_'+'valid_y',valid_y)
        np.save('/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rosaceae/chunk_folds/'+str(test_index)+'_'+'x_train',train_X)
        np.save('/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rosaceae/chunk_folds/'+str(test_index)+'_'+'y_train',train_y)
        
        
        model = spinal_cnn();
        #model = getMode();
        
        result_folder = OutputDir
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        model_results_folder=result_folder
        
        #best_weights = model_results_folder + 'best_weights.h5'
        
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30, restore_best_weights=True)
        model_check = ModelCheckpoint(filepath = OutputDir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True, save_weights_only=True)
        #reduct_L_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=20)
        reduct_L_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        cbacks = [model_check, early_stopping,reduct_L_rate]
        
        #####################Call back #########################
        #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        ########################################################
        
        history = model.fit(train_X, train_y, batch_size = 128, epochs = 60, validation_data = (valid_X, valid_y),callbacks = cbacks);
        
        ##################### EXTRA ###########################
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        #########################################################
        
        trainning_result.append(calculateScore(train_X, train_y, model, folds));
        validation_result.append(calculateScore(valid_X, valid_y, model, folds));
        testing_result.append(calculateScore(test_X, test_y, model, folds));

    temp_dict = (trainning_result, validation_result, testing_result)
    analyze(temp_dict, OutputDir);
    


PositiveCSV = 'rice_pos.txt'
NegativeCSV = 'rice_neg.txt'

OutputDir = '/home/zeeshan/SNNRice6mA-master/2nd/New/Output_Rice/'
funciton(PositiveCSV, NegativeCSV, OutputDir, 5);






