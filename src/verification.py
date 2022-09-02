from __future__ import print_function
import os
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np

def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave,sample_rate)
    return mfcc_features

def buildDataSet(dir):
    #1.albalo    2.gilas
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1]
        feature = extract_mfcc(dir+fileName)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                              [0, tmp_p, tmp_p, tmp_p , 0], \
                              [0, 0, tmp_p, tmp_p,tmp_p], \
                              [0, 0, 0, 0.5, 0.5], \
                              [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.9, 0.025, 0.025, 0.025, 0.025],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           n_iter=100, tol=0.01)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)
        GMMHMM_Models[label] = model
    return GMMHMM_Models

def main():
    trainDir = './train_audio/'
    trainDataSet = buildDataSet(trainDir)
    print("Finish prepare the training data")

    hmmModels = train_GMMHMM(trainDataSet)
    print("Finish training of the GMM_HMM models")

    testDir = './test_audio/'
    testDataSet = buildDataSet(testDir)
    score_cnt = 0
    count=0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        for i in range(0,len(feature)):
            scoreList = {}
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(feature[i])
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            count=count+1
            print("Test on true label ", label, ": predict result label is ", predict)
            if predict == label:
                score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/count), "%")





if __name__ == '__main__':
    main()