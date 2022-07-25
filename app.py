from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from tkinter import *
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from numpy import *
import tkinter as tk
from tkinter import font  as tkfont

#Main class to define the application window
class App(tk.Tk):
    # Main function being called when the code runs
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        #fonts used for Labels in the application
        self.title_font = tkfont.Font(family='Calibri', size=40, weight='bold')
        self.sub_title_font = tkfont.Font(family='Calibri', size=20)
        self.title2_font = tkfont.Font(family='Calibri', size=30, weight='bold', underline=1)
        self.small_font = tkfont.Font(family='Calibri', size=10)

        #container to stack the frames of application
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        #Defining frames for each pages of application
        self.frames = {}
        for F in (HomePage, TrainingPage, PredictionPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        #Default page of the applicaiton
        self.show_frame("HomePage")
    #method to show the requested frame in the main window
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

#Class containing the Hopempage configfuration
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Frame.configure(self, bg='azure')
        lbl0 = tk.Label(self, text="Reflected Sound Signals Discriminator", bg='azure',
                         font=controller.title_font).place(x=150, y=100)
        lbl1 = tk.Label(self, text="Do you want to train a classifier or do the prediction:", bg='azure',
                         font=controller.sub_title_font).place(x=260, y=220)
        lbl2 = tk.Label(self, text="© M.Eng Students                         Sidra Hussain 1318131           Jagdeep Singh 1317679      Vishakha Babulal 1324497", bg='azure', wraplength=170,
                        justify="center", font=controller.small_font).place(x=450, y=500)

        button1 = tk.Button(self, text="Train",
                            command=lambda: controller.show_frame("TrainingPage"), width=30, height=3)
        button2 = tk.Button(self, text="Predict",
                            command=lambda: controller.show_frame("PredictionPage"), width=30, height=3)
        button3 = tk.Button(self, text='Exit', highlightbackground='brown', bg='brown', fg='white', command=exit, width=30,
                            height=2)
        button1.place(x=210, y=300)
        button2.place(x=530, y=300)
        button3.place(x=370, y=420)

#Class containing the Training page configuration
class TrainingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Frame.configure(self, bg='azure')
        self.labelText = StringVar(self, "Training")
        self.buttonText = StringVar(self, "Start")
        lbl0 = tk.Label(self, textvariable=self.labelText, font=controller.title2_font, bg='azure', wraplength=600,justify="center").place(x=90, y=50)
        lbl1 = tk.Label(self, text='Choose the combined excel file of Object1 :', bg='azure').place(x=90, y=150)
        btn1 = tk.Button(self, text='Select', command=self.importFile1).place(x=70,
                                                                                                               y=180,
                                                                                                               width=90,
                                                                                                               height=27)
        self.en0 = Entry(self, bg="#eee")
        self.en0.place(x=170, y=180)
        lbl2 = tk.Label(self, text='Choose the combined excel file of Object2 :', bg='azure').place(x=390, y=150)
        btn2 = tk.Button(self, text='Select', command=self.importFile2).place(x=380,
                                                                                                               y=180,
                                                                                                               width=90,
                                                                                                               height=27)
        self.en1 = Entry(self, bg="#eee")
        self.en1.place(x=480, y=180)
        lbl3 = tk.Label(self, text='Choose a data file to test the trained model :', bg='azure').place(x=700, y=150)
        btn3 = tk.Button(self, text='Select', command=self.importFile3).place(x=690,
                                                                              y=180,
                                                                              width=90,
                                                                              height=27)
        self.en2 = Entry(self, bg="#eee")
        self.en2.place(x=790, y=180)
        lbl4 = tk.Label(self, text="Assessment of Classification Model:", bg='azure',
                        font=controller.sub_title_font).place(x=370, y=340)
        self.lbl5 = tk.Label(self, text='Accuracy :', bg='azure').place(x=90, y=390)
        self.en3 = Entry(self, bg="#eee")
        self.en3.place(x=170, y=390, width=320)
        self.lbl6 = tk.Label(self, text='Precision :', bg='azure').place(x=90, y=420)
        self.en4 = Entry(self, bg="#eee")
        self.en4.place(x=170, y=420, width=320)
        self.lbl7 = tk.Label(self, text='Recall :', bg='azure').place(x=90, y=450)
        self.en5 = Entry(self, bg="#eee")
        self.en5.place(x=170, y=450, width=320)
        self.lbl8 = tk.Label(self, text='F1 Score :', bg='azure').place(x=90, y=480)
        self.en6 = Entry(self, bg="#eee")
        self.en6.place(x=170, y=480, width=320)
        self.lbl9 = tk.Label(self, text='TNR :', bg='azure').place(x=550, y=390)
        self.en7 = Entry(self, bg="#eee")
        self.en7.place(x=600, y=390, width=320)
        self.lbl10 = tk.Label(self, text='FDR :', bg='azure').place(x=550, y=420)
        self.en8 = Entry(self, bg="#eee")
        self.en8.place(x=600, y=420, width=320)
        self.lbl11 = tk.Label(self, text='NPV :', bg='azure').place(x=550, y=450)
        self.en9 = Entry(self, bg="#eee")
        self.en9.place(x=600, y=450, width=320)
        self.lbl12 = tk.Label(self, text='TPR :', bg='azure').place(x=550, y=480)
        self.en10 = Entry(self, bg="#eee")
        self.en10.place(x=600, y=480, width=320)
        self.btn4 = tk.Button(self, textvariable=self.buttonText, highlightbackground='green', bg='green', fg='white', command= self.train).place(x=410, y=265, width=230,
            height=45)
        btn5 = tk.Button(self, text='Back', command=lambda: [controller.show_frame("HomePage"), self.changeText()],
                         highlightbackground='pink3').place(x=340, y=580,
                                                            width=100,
                                                            height=30)
        btn6 = tk.Button(self, text='Exit', highlightbackground='brown', bg='brown', fg='white', command=exit).place(x=600,
                                                                                                         y=580,
                                                                                                         width=100,
                                                                                                     height=30)


    #functions to import files from user
    def importFile1(self):
        self.import_file_path1 = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls")])
        self.en0.insert(END, str(self.import_file_path1))

    def importFile2(self):
        self.import_file_path2 = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls")])
        self.en1.insert(END, str(self.import_file_path2))

    def importFile3(self):
        self.import_file_path3 = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls")])
        self.en2.insert(END, str(self.import_file_path3))

    #Function to change label text for re-training
    def changeText(self):
        self.labelText.set("Training")

    #function to clear entries for re-training
    def tryAgain(self):
        self.en0.delete(0, 'end')
        self.en1.delete(0, 'end')
        self.en2.delete(0, 'end')
        self.en3.delete(0, 'end')
        self.en4.delete(0, 'end')
        self.en5.delete(0, 'end')
        self.en6.delete(0, 'end')
        self.en7.delete(0, 'end')
        self.en8.delete(0, 'end')
        self.en9.delete(0, 'end')
        self.en10.delete(0, 'end')
        self.btn = tk.Button(self, textvariable=self.buttonText, highlightbackground='green', bg='green', fg='white',
                              command=self.train).place(x=410, y=265, width=230,
                                                        height=45)
        self.changeText()

    def changeTextButton(self):
        self.btn = tk.Button(self, text="Try Again", highlightbackground='green', bg='green', fg='white', command= self.tryAgain).place(x=410, y=265, width=230,
            height=45)

    #Function copies the imported files in local directory and pass the control to main.py script to do the training of model
    def train(self):
        import joblib
        import pandas as pd
        import numpy as np
        import math

        import matplotlib.pyplot as plt


        print("LOADING OBJECT DATA FOR MODEL TRAINING")
        # Importing Object Data, deleting first incremental index row
        Reflected_Signals_Object_1 = pd.read_excel(self.import_file_path1, header=None)
        Reflected_Signals_Object_2 = pd.read_excel(self.import_file_path2, header=None)
        print(Reflected_Signals_Object_1.shape)
        print(Reflected_Signals_Object_1.shape)
        Reflected_Signals_Object_1_Array = np.array(Reflected_Signals_Object_1)
        Reflected_Signals_Object_1_Array = np.delete(Reflected_Signals_Object_1_Array, 0, 0)
        Reflected_Signals_Object_2_Array = np.array(Reflected_Signals_Object_2)
        Reflected_Signals_Object_2_Array = np.delete(Reflected_Signals_Object_2_Array, 0, 0)
        print("\nLOADING DATA COMPLETE")

        # Features Definitions, Peak time, RMS energy, Average Energy, Maximum Loudness, Total Energy

        # RMS Energy of the signal
        def rms(spec, t):

            sum = 0.0
            for i in range(1, len(spec)):

                for j in range(1, len(t)):
                    sum = sum + spec[i][j] ** 2.0

                sum = sum / (1.0 * len(spec))

            return math.sqrt(sum)

        # Average Energy of the signal
        def Average(lst):
            return sum(lst) / len(lst)

            # Maximum Loudness/ Energy of the signal

        def loudness(spec):
            return np.amax(abs(spec[0]))

        # Frequency at peak or Maximum Loudness/Energy
        def peak_f(spec, f):
            return f[np.argmax(abs(spec[0]))]

        # Time at peak or Maximum Loudness/Energy
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

            plt.figure()
            for i in range(data_array.shape[0]):
                spectrogram, f, t, im = plt.specgram(
                    data_array[i], NFFT=256, Fs=2, noverlap=0);

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

        # Feature extraction of object 1 signals
        rms1 = []
        loud1 = []
        pkf1 = []
        pkt1 = []
        av1 = []
        sum1 = []
        rms1, loud1, pkf1, pkt1, av1, sum1 = feature_extraction(Reflected_Signals_Object_1_Array)

        # Feature extraction of object 2 signals
        rms2 = []
        loud2 = []
        pkf2 = []
        pkt2 = []
        av2 = []
        sum2 = []
        rms2, loud2, pkf2, pkt2, av2, sum2 = feature_extraction(
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

        # Checking sizes of object signal arrays
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

        # Targets Assignment
        sum_obj1["Target"] = str('Object 1')
        sum_obj2["Target"] = str('Object 2')

        # Accumulation of data Frames of features
        pk_f = [pk_freq_obj1, pk_freq_obj2]

        rootms = [rms_obj1, rms_obj2]

        max_sum = [sum_obj1, sum_obj2]

        frame_avg = [avg_obj1, avg_obj2]

        pk_t = [pk_time_obj1, pk_time_obj2]

        L = [loud_obj1, loud_obj2]

        final_pkfreq = pd.concat(pk_f)
        final_rms = pd.concat(rootms)
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
        final_data_frame1 = pd.concat([final_pkfreq, final_pkt], axis=1)
        final_data_frame2 = pd.concat([final_data_frame1, final_rms], axis=1)
        final_data_frame3 = pd.concat([final_data_frame2, final_avg], axis=1)
        final_data_frame4 = pd.concat([final_data_frame3, final_l], axis=1)
        final_data_frame = pd.concat([final_data_frame4, final_max_sum], axis=1)
        final_data_frame.isnull().values.any()
        print(final_data_frame.shape)
        print("\nDATA FRAME COMPLETE")

        # Splitting training and testing of data frame, 20% for testing
        print("\nSPLITTING TRAINING AND TESTING DATA, USING 20% FOR TESTING")
        X = final_data_frame.drop("Target", axis=1)
        y = final_data_frame["Target"]
        np.random.seed(42)

        print(X.shape)

        print(y.shape)

        # Split in train and test set,20 % data to be used for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)

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
            # Dictionary of model scores
            model_scores = {}
            # Each model turn
            for name, model in models.items():
                # Fitting the model
                model.fit(X_train, y_train)
                # Score appending to model
                model_scores[name] = model.score(X_test, y_test) * 100
            return model_scores

        # Classification process with plotting accuracies and classification results
        print("\nMODEL FIT COMPLETE")
        model_scores = modelfitWithscore(models, X_train, X_test, y_train, y_test)

        print("\nTHE MODEL CONFIDENCES ARE AS FOLLOWS:")
        print(model_scores)

        # Bar Graph: Decision Tree and Random Forest confidence
        plt.figure()
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

        # Testing classifier with one file
        # Read test file, save into array
        # Save into array for feature extraction
        test_data = pd.read_excel(self.import_file_path3)
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

        # Data frame for the features of test file
        peak_freq = pd.DataFrame([PKF], columns=["Peak Frequency"])
        peak_time = pd.DataFrame([PKT], columns=["Propagation Delay"])
        rms_obj = pd.DataFrame([RMS], columns=["RMS Energy"])
        avg_obj = pd.DataFrame([AV], columns=["Average Energy"])
        loud_obj = pd.DataFrame([LOUD], columns=["Maximum Loudness"])
        sum_obj = pd.DataFrame([SUM], columns=["Total Energy"])
        frames1 = [peak_freq, peak_time]
        final_data_frame1 = pd.concat(frames1, axis=1)
        final_data_frame2 = pd.concat([final_data_frame1, rms_obj], axis=1)
        final_data_frame3 = pd.concat([final_data_frame2, avg_obj], axis=1)
        final_data_frame0 = pd.concat([final_data_frame3, loud_obj], axis=1)
        final_data_frame01 = pd.concat([final_data_frame0, sum_obj], axis=1)
        print("\nFINAL DATA FRAME OF TEST SIGNAL MADE SUCCESSFULLY.\n")
        print(final_data_frame01)

        # Classifier Analysis
        print("\nUSING SAVED MODEL RANDOM FOREST")
        loaded_model = joblib.load(trained_model1)
        value = loaded_model.predict(final_data_frame01)
        print("\nThe object is:")
        print(value)

        print("\nSTARTING CLASSIFIER ASSESSMENT")
        # Confusion Matrix of the classification
        import pandas as pd
        y_actu = y_test
        y_pred = models["Random Forest"].predict(X_test)


        # series to array
        w = y_actu.tolist()
        y_actu0 = np.array(w)

        y_actu = pd.Series(y_actu0, name='Actual')
        y_pred = pd.Series(y_pred, name='Predicted')

        # data frame of Confusion Matrix
        df_confusion = pd.crosstab(y_actu, y_pred)

        # Confusion Matrix
        print("\nConfusion Matrix of the classifier:\n")
        print(df_confusion)

        # confusion matrix with sum of rows and columns
        df_confusion_all = pd.crosstab(y_actu, y_pred,
                                       rownames=['Actual'],
                                       colnames=['Predicted'],
                                       margins=True)

        print("\nConfusion Matrix of the classifier with sums:\n")
        print(df_confusion_all)

        # confusion matrix normalized
        df_conf_norm = df_confusion / df_confusion.sum(axis=1)
        print("\nNormalized Confusion Matrix of the classifier:\n")
        print(df_conf_norm)

        # confusion matrix diagram block
        def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
            plt.matshow(df_confusion, cmap=cmap)  # imshow
            # plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(df_confusion.columns))
            plt.xticks(tick_marks, df_confusion.columns, rotation=45)
            plt.yticks(tick_marks, df_confusion.index)
            # plt.tight_layout()
            plt.ylabel(df_confusion.index.name)
            plt.xlabel(df_confusion.columns.name)

        plot_confusion_matrix(df_confusion)

        plot_confusion_matrix(df_conf_norm)

        # true negative
        TN = df_confusion.iat[0, 0]
        # true positive
        TP = df_confusion.iat[1, 1]
        # false positive
        FP = df_confusion.iat[0, 1]
        # false negative
        FN = df_confusion.iat[1, 0]

        # Classifier Assessment rates

        # Sensitivity/ Recall/ Hit rate
        TPR = TP / (TP + FN)
        print("\nTrue Positive Rate/Sensitivity/Recall/Hit Rate:")
        print(TPR)

        # Specificity/ Selectivity
        TNR = TN / (TN + FP)
        print("\nTrue Negative Rate/Specificty/Selectivity:")
        print(TNR)

        # False Discovery Rate
        FDR = FP / (FP + TP)
        print("\nFalse Discovery Rate:")
        print(FDR)

        # Negative Predictive Value
        NPV = TN / (TN + FN)
        print("\nNegative Predictive Value:")
        print(NPV)

        ACCURACY = (TP + TN) / (TP + TN + FP + FN)

        CLASSIIFIER_ANALYSIS = {}
        CLASSIIFIER_ANALYSIS["TPR"] = TPR
        CLASSIIFIER_ANALYSIS["TNR"] = TNR
        CLASSIIFIER_ANALYSIS["FDR"] = FDR
        CLASSIIFIER_ANALYSIS["NPV"] = NPV

        CA_DATAFRAME = pd.DataFrame(CLASSIIFIER_ANALYSIS, index=["Classifier Analysis Rates"])
        CA_DATAFRAME.T.plot.bar(rot=0);

        # Recall
        RECALL = TPR

        # Precision
        PRECISION = TP / (TP + FP)

        # F1 score
        F1_SCORE = 2 / ((1 / RECALL) + (1 / PRECISION))
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

        #Method to change the text label after completion of training
        def changeText2():
            self.labelText.set("Models are trained and saved to local disk")
            self.btn7 = tk.Button(self, text='Prediction', highlightbackground='green', bg='green', fg='white',
                             command=lambda: [self.controller.show_frame("PredictionPage"), self.changeText]).place(
                x=800,
                y=65,
                width=100,
                height=30)

        changeText2()

        #Prining values for classisfication assessment in GUI
        self.en3.insert(END, str(ACCURACY))
        self.en4.insert(END, str(PRECISION))
        self.en5.insert(END, str(RECALL))
        self.en6.insert(END, str(F1_SCORE))
        self.en7.insert(END, str(TNR))
        self.en8.insert(END, str(FDR))
        self.en9.insert(END, str(NPV))
        self.en10.insert(END, str(TPR))

        self.changeTextButton()


#Class containing the prediction page configuration
class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Frame.configure(self, bg='azure')
        self.v = IntVar()
        lbl0 = tk.Label(self, text='Prediction', font=controller.title2_font, bg='azure').place(x=90, y=50)
        lbl1 = tk.Label(self, text='Choose a file:', bg='azure').place(x=100, y=160)
        btn1 = tk.Button(self, text='Select', command=self.importFile).place(x=200, y=160, width=90, height=27)
        self.en1 = tk.Entry(self, bg="#eee")
        self.en1.place(x=300, y=160)
        lbl2 = tk.Label(self, text='Choose a Classifier:', bg='azure').place(x=100, y=210)
        self.r1 = tk.Radiobutton(self, text='Decision Tree', variable=self.v, value=1, bg='azure').place(x=300, y=210)
        self.r2 = tk.Radiobutton(self, text='Random Forest', variable=self.v, value=2, bg='azure').place(x=300, y=235)
        lbl3 = tk.Label(self, text='Enter Row Number:', bg='azure').place(x=100, y=280)
        self.en2 = tk.Entry(self, bg="#eee")
        self.en2.place(x=300, y=280)
        lbl4 = tk.Label(self, text='Enter Starting Column Number:', bg='azure').place(x=100, y=330)
        self.en3 = tk.Entry(self, bg="#eee")
        self.en3.place(x=300, y=330)
        lbl5 = tk.Label(self, text='Enter Signal Length:', bg='azure').place(x=100, y=380)
        self.en4 = tk.Entry(self, bg="#eee")
        self.en4.place(x=300, y=380)
        lbl6 = tk.Label(self, text='The reflected sound signal belongs to:', bg='azure').place(x=640, y=250)
        self.en5 = tk.Entry(self, bg="#eee")
        self.en5.place(x=670, y=300)
        btn2 = tk.Button(self, text='Predict', highlightbackground='green', bg='green', fg='white', command=self.predict).place(
            x=670,
            y=450,
            width=200,
            height=30)
        btn3 = tk.Button(self, text="Reset", highlightbackground='gold2', bg='gold2', command=self.reset).place(x=100, y=450, width=100, height=30)
        btn4 = tk.Button(self, text='Back', command=lambda: controller.show_frame("HomePage"), highlightbackground='pink3', bg='pink3').place(x=230, y=450,
                                                                                                         width=100,
                                                                                                         height=30)
        btn5 = tk.Button(self, text='Exit', highlightbackground='brown', bg='brown', fg='white', command=exit).place(x=390,
                                                                                                         y=450,
                                                                                                         width=100,
                                                                                                         height=30)
    # Method to reset the entries
    def reset(self):
        self.en1.delete(0, 'end')
        self.en2.delete(0, 'end')
        self.en3.delete(0, 'end')
        self.en4.delete(0, 'end')
        self.en5.delete(0, 'end')
        #self.r1.deselect()
        #self.r2.set(None)

    # Method to import test data file
    def importFile(self):
        self.import_file_path = filedialog.askopenfilename()
        self.en1.insert(END, str(self.import_file_path))

    # Method to choose the classification model
    def chooseModel(self):
        if (self.v.get() == 1):
            self.loaded_model = joblib.load('decisionTree.sav')
        if (self.v.get() == 2):
            self.loaded_model = joblib.load('randomForest.sav')
        print("Classification model chosen successfully")
        return self.loaded_model

    # Method to predict the time samples from imported test data file
    def predict(self):
        rowNumber = 0;
        startColNumber = 0;
        # endColumnNumber = 16384
        if (int(self.en2.get())) >= 0:
            rowNumber = int(self.en2.get())
        if (int(self.en3.get()) > 0 and int(self.en3.get()) <= 16384):
            startColNumber = int(self.en3.get())
        if (int(self.en4.get()) > 0 and int(self.en4.get()) <= 16384):
            signalLength = int(self.en4.get())
        print("RowNumber :", rowNumber)
        print("First ColumnNumber :", startColNumber)
        print("Last ColumnNumber :", signalLength)
        endColumnNumber = signalLength + startColNumber
        numberOfSamples = len(list(range(startColNumber, endColumnNumber)))
        print("Number of samples:", numberOfSamples)
        file = pd.read_excel(self.import_file_path, header=None, usecols=list(range(startColNumber, endColumnNumber)))
        print("file reading successful")
        array_test = np.array(file)
        spectrum, freqs, t, im = plt.specgram(array_test[rowNumber], NFFT=256, Fs=2, noverlap=0)

        def rms(spec, t):
            sum = 0.0
            for i in range(1, len(spec)):
                for j in range(1, len(t)):
                    sum = sum + spec[i][j] ** 2.0
                    sum = sum / (1.0 * len(spec))
            return math.sqrt(sum)

        def Average(lst):
            return sum(lst) / len(lst)

        def loudness(spec):
            return np.amax(abs(spec[0]))

        def peak_f(spec, f):
            return f[np.argmax(abs(spec[0]))]

        def peak_t(spec, t):
            return t[np.argmax(abs(spec[0]))]

        # Feature extraction of from test data file
        test_loud = loudness(spectrum)
        test_max_sum = np.sum(spectrum)
        test_avg = Average(spectrum)
        test_rms = rms(spectrum, t)
        test_pkt = peak_t(spectrum, t)
        test_pkf = peak_f(spectrum, freqs)

        # Creation of data frames from test data file
        pk_freq = pd.DataFrame([test_pkf], columns=["Peak Frequency"])
        pk_time = pd.DataFrame([test_pkt], columns=["Propagation Delay"])
        max_sum = pd.DataFrame([test_max_sum], columns=["Total Energy"])
        rms_obj = pd.DataFrame([test_rms], columns=["RMS Energy"])
        avg_obj = pd.DataFrame([test_avg], columns=["Average Energy"])
        loud_obj = pd.DataFrame([test_loud], columns=["Maximum Loudness"])
        frames = [pk_freq, pk_time]
        final_data_frame1 = pd.concat(frames, axis=1)
        final_data_frame2 = pd.concat([final_data_frame1, rms_obj], axis=1)
        final_data_frame3 = pd.concat([final_data_frame2, avg_obj], axis=1)
        final_data_frame0 = pd.concat([final_data_frame3, loud_obj], axis=1)
        final_data_frame = pd.concat([final_data_frame0, max_sum], axis=1)
        print(final_data_frame)

        # Loading the trained classification model

        self.loaded_model = self.chooseModel()

        # prediction
        result = self.loaded_model.predict(final_data_frame)
        final_prediction = "Object-1"
        if result[0] != "Object 1":
            final_prediction = "Object-2"
        self.en5.delete(0, 'end')
        self.en5.insert(END, str(final_prediction))


if __name__ == "__main__":
    app = App()
    app.title('Reflected Sound Signal Discriminator')
    app.geometry("1050x650+10+10")
    image1 = Image.open("logo.png")
    test = ImageTk.PhotoImage(image1)
    label1 = tk.Label(image=test, width=130, height=40, bg='azure')
    label1.image = test
    label1.place(x=450, y=580)
    app.mainloop()
