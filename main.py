
import joblib
import pandas as pd
import numpy as np
import math
from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

print("LOADING OBJECT DATA FOR MODEL TRAINING")
#Importing Object Data, deleting first incremental index row
Reflected_Signals_Object_1 = pd.read_excel('Object1.xlsx', header = None)
Reflected_Signals_Object_2 = pd.read_excel('Object2.xlsx', header = None)
print(Reflected_Signals_Object_1.shape)
print(Reflected_Signals_Object_1.shape)
Reflected_Signals_Object_1_Array = np.array(Reflected_Signals_Object_1)
Reflected_Signals_Object_1_Array = np.delete(Reflected_Signals_Object_1_Array, 0, 0)
Reflected_Signals_Object_2_Array = np.array(Reflected_Signals_Object_2)
Reflected_Signals_Object_2_Array = np.delete(Reflected_Signals_Object_2_Array, 0, 0)
print("\nLOADING DATA COMPLETE")

#Features Definitions, Peak time, RMS energy, Average Energy, Maximum Loudness, Total Energy

#RMS Energy of the signal
def rms(spec, t):
       
        sum = 0.0
        for i in range(1, len(spec)):
            
            for j in range (1, len(t)):
                sum = sum + spec[i][j] ** 2.0

            sum = sum / (1.0 * len(spec))

        return math.sqrt(sum)

#Average Energy of the signal
def Average(lst):
    return sum(lst) / len(lst)   

#Maximum Loudness/ Energy of the signal
def loudness(spec):
    return np.amax(abs(spec[0]))

#Frequency at peak or Maximum Loudness/Energy
def peak_f(spec, f):
    return f[np.argmax(abs(spec[0]))]

#Time at peak or Maximum Loudness/Energy
def peak_t(spec, t):
    return t[np.argmax(abs(spec[0]))]

# Features extraction from object 1
def feature_extraction(data_array):
    sumenergy = []

    av0 = 0
    av = []

    rms0 = 0
    Rms = []

    loud0 = 0
    loud = []

    pkf0 = 0
    pkf = []

    pkt0 = 0
    pkt = []

    for i in range(data_array.shape[0]):

        spectrogram, f, t, im = plt.specgram(
            data_array[i],NFFT=256, Fs=2, noverlap=0);

        sumenergy.append(np.sum(spectrogram))

        rms0 = rms(spectrogram, t)
        Rms.append(rms0)

        loud0 = loudness(spectrogram)
        loud.append(loud0)

        pkf0 = peak_f(spectrogram, f)
        pkf.append(pkf0)

        pkt0 = peak_t(spectrogram, t)
        pkt.append(pkt0)

        av0 = Average(spectrogram)
        av.append(av0)



    return Rms, loud, pkf, pkt, av, sumenergy


print("\nCALCULATING SPECTROGRAMS AND EXTRACTING FEATURES FROM OBJECTS SIGNALS")

#Feature extraction of object 1 signals
rms1 = []
loud1 = []
pkf1 = []
pkt1 = []
av1 = []
sum1 = []
rms1, loud1, pkf1, pkt1, av1, sum1  = feature_extraction(
    Reflected_Signals_Object_1_Array)

#Feature extraction of object 2 signals
rms2 = []
loud2 = []
pkf2 = []
pkt2 = []
av2 = []
sum2 = []
rms2, loud2, pkf2, pkt2, av2, sum2  = feature_extraction(
    Reflected_Signals_Object_2_Array)

print("\nCREATING DATA FRAME MODEL")
# Merging and creating data frame

pk_freq_obj1 = pd.DataFrame(pkf1, columns=["Peak Frequency"])
pk_freq_obj2 = pd.DataFrame(pkf2, columns=["Peak Frequency"])

pk_time_obj1 = pd.DataFrame(pkt1, columns=["Propagation Delay"])
pk_time_obj2 = pd.DataFrame(pkt2, columns=["Propagation Delay"])

sum_obj1 = pd.DataFrame(sum1, columns=["Total Energy"])
sum_obj2 = pd.DataFrame(sum2, columns=["Total Energy"])

rms_obj1 = pd.DataFrame(rms1, columns=["RMS Energy"])
rms_obj2 = pd.DataFrame(rms2, columns=["RMS Energy"])

avg_obj1 = pd.DataFrame(av1, columns=["Average Energy"])
avg_obj2 = pd.DataFrame(av2, columns=["Average Energy"])

loud_obj1 = pd.DataFrame(loud1, columns=["Maximum Loudness"])
loud_obj2 = pd.DataFrame(loud2, columns=["Maximum Loudness"])

#Checking sizes of object signal arrays
print(pk_freq_obj1.shape)
print(pk_freq_obj2.shape)
print(sum_obj1.shape)
print(sum_obj2.shape)
print(rms_obj1.shape)
print(rms_obj2.shape)
print(avg_obj1.shape)
print(avg_obj2.shape)
print(pk_time_obj1.shape)
print(pk_time_obj2.shape)
print(loud_obj1.shape)
print(loud_obj2.shape)

#Targets Assignment
sum_obj1["Target"] = str('Object 1') 
sum_obj2["Target"] = str('Object 2')

#Accumulation of data Frames of features
pk_f = [pk_freq_obj1, pk_freq_obj2]

rootms = [rms_obj1, rms_obj2]

max_sum = [sum_obj1, sum_obj2]

frame_avg = [avg_obj1, avg_obj2]

pk_t = [pk_time_obj1, pk_time_obj2]

L = [loud_obj1, loud_obj2]

final_pkfreq = pd.concat(pk_f)
final_rms = pd. concat(rootms)
final_max_sum = pd.concat(max_sum)
final_avg = pd.concat(frame_avg)
final_pkt = pd.concat(pk_t)
final_l = pd.concat(L)
print(final_max_sum.shape)
print(final_pkfreq.shape)
print(final_rms.shape)
print(final_avg.shape)


# Final Data Frame :
# Peak freq, Propagation time
# RMS energy, Average Energy
# Maximum Loudness, Total Energy
final_data_frame1 = pd.concat([final_pkfreq, final_pkt], axis = 1)
final_data_frame2 = pd.concat([final_data_frame1, final_rms], axis = 1)
final_data_frame3 = pd.concat([final_data_frame2, final_avg], axis = 1)
final_data_frame4 = pd.concat([final_data_frame3, final_l], axis = 1)
final_data_frame = pd.concat([final_data_frame4, final_max_sum], axis =1)
final_data_frame.isnull().values.any()
print(final_data_frame.shape)
print("\nDATA FRAME COMPLETE")

# Splitting training and testing of data frame, 20% for testing
print("\nSPLITTING TRAINING AND TESTING DATA, USING 20% FOR TESTING")
X = final_data_frame.drop("Target", axis =1)
y = final_data_frame["Target"]
np.random.seed(42)

print(X.shape)

print(y.shape)

# Split in train and test set,20 % data to be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2)

# Save models in the dictionary
models = {"Random Forest": RandomForestClassifier(),
         "Decision Tree": tree.DecisionTreeClassifier()}


# Create a function to fit and score models
# fits and evaluates the machine learning models
# X_train: training data(no labels)
# X_test: testing data(no labels)
# y_train: training labels
# y_test: testing labels
def modelfitWithscore(models, X_train, X_test, y_train, y_test):
    # Randomizing the data frame
    np.random.seed(42)
    #Dictionary of model scores
    model_scores = {}
    #Each model turn
    for name, model in models.items():
        # Fitting the model
        model.fit(X_train, y_train)
        # Score appending to model
        model_scores[name] = model.score(X_test, y_test)*100
    return model_scores

#Classification process with plotting accuracies and classification results
print("\nMODEL FIT COMPLETE")
model_scores = modelfitWithscore(models, X_train,X_test, y_train, y_test)

print("\nTHE MODEL CONFIDENCES ARE AS FOLLOWS:")
print(model_scores)

#Bar Graph: Decision Tree and Random Forest confidence
plotdata = pd.DataFrame(
    {"Random Forest": list(model_scores.items())[0][1], "Decision Tree": list(model_scores.items())[1][1]},
    index=["Accuracies"])
plotdata.plot(kind="barh")


# Trained Model Saved to local machine
trained_model1 = 'randomForest.sav'
trained_model2 = 'decisionTree.sav'
print("\nTRAINED MODELS SAVED TO LOCAL MACHINE")
joblib.dump(models["Random Forest"], trained_model1)
joblib.dump(models["Decision Tree"], trained_model2)

#Testing classifier with one file
#Read test file, save into array
#Save into array for feature extraction
test_data = pd.read_excel('test_obj.xlsx')
array_test = np.array(test_data)
print("\nLOADED TEST DATA ARRAY FOR OBJECT DISCRIMINATION AND CLASSIFIER ASSESSMENT")

RMS = []
LOUD = []
PKF = []
PKT = []
AV = []
SUM = []
RMS, LOUD, PKF, PKT, AV, SUM = feature_extraction(array_test)
print("\nSPECTROGRAMS CALCULATED AND FEATURES EXTRACTED OF TEST SIGNAL")
print("\nFEATURES: Peak Frequency, Propagation Delay, RMS Energy, Average Energy and Maximum Loudness")


#Data frame for the features of test file
peak_freq = pd.DataFrame([PKF], columns=["Peak Frequency"])
peak_time = pd.DataFrame([PKT], columns= ["Propagation Delay"])
rms_obj = pd.DataFrame([RMS], columns=["RMS Energy"])
avg_obj = pd.DataFrame([AV], columns=["Average Energy"])
loud_obj = pd.DataFrame([LOUD], columns=["Maximum Loudness"])
sum_obj = pd.DataFrame([SUM], columns=["Total Energy"])
frames1 = [peak_freq, peak_time]
final_data_frame1 = pd.concat(frames1, axis =1)
final_data_frame2 = pd.concat([final_data_frame1,rms_obj], axis =1)
final_data_frame3 = pd.concat([final_data_frame2,avg_obj], axis =1)
final_data_frame0 = pd.concat([final_data_frame3,loud_obj], axis =1)
final_data_frame01 = pd.concat([final_data_frame0,sum_obj], axis =1)
print("\nFINAL DATA FRAME OF TEST SIGNAL MADE SUCCESSFULLY.\n")
print(final_data_frame01)

#Classifier Analysis
print("\nUSING SAVED MODEL RANDOM FOREST")
loaded_model = joblib.load(trained_model1)
value = loaded_model.predict(final_data_frame01)
print("\nThe object is:")
print(value)


print("\nSTARTING CLASSIFIER ASSESSMENT")
#Confusion Matrix of the classification
import pandas as pd
y_actu = y_test
y_pred = models["Random Forest"].predict(X_test)

#series to array
w=y_actu.tolist()
y_actu0=np.array(w)

y_actu = pd.Series(y_actu0, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')

#data frame of Confusion Matrix
df_confusion = pd.crosstab(y_actu, y_pred)

# Confusion Matrix
print("\nConfusion Matrix of the classifier:\n")
print(df_confusion)

#confusion matrix with sum of rows and columns
df_confusion_all = pd.crosstab(y_actu, y_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'],
                               margins=True)

print("\nConfusion Matrix of the classifier with sums:\n")
print(df_confusion_all)

#confusion matrix normalized
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
print("\nNormalized Confusion Matrix of the classifier:\n")
print(df_conf_norm)

#confusion matrix diagram block
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)

plot_confusion_matrix(df_conf_norm)

#true negative
TN = df_confusion.iat[0,0]
# true positive
TP = df_confusion.iat[1,1]
# false positive
FP = df_confusion.iat[0,1]
# false negative
FN = df_confusion.iat[1,0]

#Classifier Assessment rates

#Sensitivity/ Recall/ Hit rate
TPR = TP/(TP + FN)
print("\nTrue Positive Rate/Sensitivity/Recall/Hit Rate:")
print(TPR)

#Specificity/ Selectivity
TNR = TN/(TN + FP)
print("\nTrue Negative Rate/Specificty/Selectivity:")
print(TNR)

#False Discovery Rate
FDR = FP/(FP + TP)
print("\nFalse Discovery Rate:")
print(FDR)

#Negative Predictive Value
NPV = TN/(TN + FN)
print("\nNegative Predictive Value:")
print(NPV)


ACCURACY = (TP + TN)/(TP + TN + FP + FN)

CLASSIIFIER_ANALYSIS = {}
CLASSIIFIER_ANALYSIS["TPR"] = TPR
CLASSIIFIER_ANALYSIS["TNR"] = TNR
CLASSIIFIER_ANALYSIS["FDR"] = FDR
CLASSIIFIER_ANALYSIS["NPV"] = NPV

CA_DATAFRAME = pd.DataFrame(CLASSIIFIER_ANALYSIS, index=["Classifier Analysis Rates"])
CA_DATAFRAME.T.plot.bar(rot = 0);

#Recall
RECALL = TPR

#Precision
PRECISION = TP / (TP + FP)

# F1 score
F1_SCORE= 2/((1/RECALL)+(1/PRECISION))
print("\nRecall of the classifier is:")
print(RECALL)
print("\nPrecision of the classifier is:")
print(PRECISION)
print("\nF1-Score of the classifier is:")
print(F1_SCORE)

# Plotting ROC curve
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

metrics.plot_roc_curve(models["Random Forest"], X_test, y_test)  
plt.show()

