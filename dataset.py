import glob
import pandas as pd


Option= input("A default Object1 test_obj is already present in code directory, to make your own test_obj:"
              "\nEnter 'T' or 't' for converting csv test file to excel file, for tes_obj.xlsx of your choice."
              "\n OR"
              "\nEnter any other letter to merge and convert csv files of objects to excel for Object1.xlsx and Object2.xlsx: ")
if Option == 'T' or Option == 't':
    # CSV TO EXCEL OF TEST-FILE test_obj USED IN MAIN.PY AND APP.PY FOR CLASSIFIER TEST AND ASSESSMENT
    # User enters path of csv task file of any object as per wish.
    # The test of classifier in main.py reads this file for its test.
    OBJECTTESTpath = input("For testing signal, "
                           "enter path of csv file of task 2 any object of wish "
                           "for e.g path_to_csv_files\Object 1\side1-0.5\data\ "
                           "(do not forget \ in the end): ")
    print(OBJECTTESTpath)
    filenametest = input("Enter any signal number of wish, for e.g. 001 or 023 till 030: ")
    print(filenametest)
    filenamestest = (OBJECTTESTpath + filenametest +'.csv')
    dftest = []
    print(filenamestest)
    CSV_FILE2 = pd.read_csv(filenamestest, header=None, usecols=list(range(16384)))
    dftest.append(CSV_FILE2)
    frametest = pd.concat(dftest, axis=0, ignore_index=True)
    frametest.to_excel(r'test_obj.xlsx', index=None)
    print("test_obj excel file ready")

# MERGING OBJECT 1 SIGNAL FILE


# User enters path of csv task files object 1 folder
OBJECT1path = input("Enter path of csv files of task 2 object 1 folder for e.g path_to_csv_files\Object 1: ")
print(OBJECT1path)
filenames1 = glob.glob(OBJECT1path + '\*\*\*.csv')
df1 = []
for filename in filenames1:
    print(filename)
    CSV_FILE1 = pd.read_csv(filename, header = None, usecols=list(range(16384)))
    df1.append(CSV_FILE1)
frame1 = pd.concat(df1, axis=0, ignore_index=True)
frame1.to_excel(r'Object1.xlsx', index=None)
print("Object 1 Data Merge Complete")


# # MERGING OBJECT 2 SIGNAL FILE
# User enters path of csv task files object 2 folder
OBJECT2path = input("Enter path of csv files of task 2 object 2 folder for e.g path_to_csv_files\Object 2: ")
print(OBJECT2path)


filenames2 = glob.glob(OBJECT2path + '\*\*\*.csv')
df2 = []
for filename in filenames2:
    print(filename)
    CSV_FILE2 = pd.read_csv(filename, header = None, usecols=list(range(16384)))
    df2.append(CSV_FILE2)
frame2 = pd.concat(df2, axis=0, ignore_index=True)
frame2.to_excel(r'Object2.xlsx', index=None)
print("Object 2 Data Merge Complete")


