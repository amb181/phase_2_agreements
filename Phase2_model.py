import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os, pickle, re, glob, sys, pymysql
from sklearn.feature_extraction.text import TfidfVectorizer # Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics

## Create a folder with your agreement type as name + "_model", ex. "NDA_model" and under this a folder called "Detected"
## Place the agreements to be processed in txt format under a folder called as your agreement type ex. "NDA"
## Finally, only change the agreement_type at the very beggining of the code ex. "NDA"

agreement_type = "NDA"
model_path = "/home/ealloem/Documents/Phase2_agreements/" + agreement_type + "_model/"
new_data_path = "/home/ealloem/Documents/Phase2_agreements/" + agreement_type + "/"


def Train_Model():
    # Check if already trained
    file1 = model_path + agreement_type + "_classes.sav"
    file2 = model_path + agreement_type + "_svc_model.sav"
    file3 = model_path + agreement_type + "_calibrated_svc.sav"
    file4 = model_path + agreement_type + "_vectorizer_fit.sav"

    if os.path.isfile(file1) and os.path.isfile(file2) and os.path.isfile(file3) and os.path.isfile(file4):
        print("Already trained.")
        classes = pickle.load(open(os.path.join(model_path + agreement_type + "_classes.sav"), 'rb'))
        svc_model = pickle.load(open(model_path + agreement_type + "_svc_model.sav", 'rb'))
        calibrated_svc = pickle.load(open(model_path + agreement_type + "_calibrated_svc.sav", 'rb'))
        tfidf = pickle.load(open(model_path + agreement_type + "_vectorizer_fit.sav", 'rb'))
    else:
        print("Training...")
        # Open dataset
        df = pd.read_csv(model_path + agreement_type + '.csv')
        # Check for missing values
        print(df.isnull().sum())
        # Output summary about the dataframe
        print(df.info())
        # Get different sections (classes)
        print(df['Section'].nunique())

        # # Plot number of examples per section
        # fig = plt.figure(figsize=(8,6))
        # df.groupby('Section').Paragraph.count().plot.bar(ylim=0)
        # plt.ylabel('Times in dataset')
        # plt.xlabel('Section')
        # plt.show()

        # Vectorize text data
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 3), stop_words='english')
        features = tfidf.fit_transform(df['Paragraph'].values.astype('U')).toarray()
        labels = df['Category_ID']
        features.shape

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

        # For large datasets consider using LinearSVC or SGDClassifier instead
        svc_model = LinearSVC()
        svc_model.fit(X_train, y_train)

        # Wrap model to get probabilities in prediction
        calibrated_svc = CalibratedClassifierCV(svc_model, method='sigmoid', cv=3)
        calibrated_svc.fit(X_train, y_train)

        # Predict
        y_pred = svc_model.predict(X_test)

        # # Calculate confussion matrix
        # conf_mat = confusion_matrix(y_test, y_pred, normalize = 'true')
        # sns.heatmap(conf_mat, annot=True, fmt='.2f', xticklabels=df[['Section', 'Category_ID']].drop_duplicates().Section.values, yticklabels=df[['Section', 'Category_ID']].drop_duplicates().Section.values)
        # plt.ylabel('Class')
        # plt.xlabel('Predicted')
        # plt.show()
        
        # Get metrics per class
        print(metrics.classification_report(y_test, y_pred, target_names=df['Section'].unique()))

        # save the model and variables to disk
        classes = dict(df[['Category_ID', 'Section']].drop_duplicates().values)
        pickle.dump(classes, open(os.path.join(model_path + agreement_type + "_classes.sav"), 'wb'))
        pickle.dump(svc_model, open(os.path.join(model_path + agreement_type + "_svc_model.sav"), 'wb'))
        pickle.dump(calibrated_svc, open(os.path.join(model_path + agreement_type + "_calibrated_svc.sav"), 'wb'))
        pickle.dump(tfidf, open(os.path.join(model_path + agreement_type + "_vectorizer_fit.sav"), 'wb'))

    return classes, svc_model, calibrated_svc, tfidf


def Predict(agreement, classes, svc_model, calibrated_svc, tfidf):
    # Open agreement
    with open(new_data_path + agreement, encoding="utf-8") as f:
        content = f.read()

    # Sections to save into DB
    sections = [''] * len(classes)

    # Read agreement line by line and build paragraphs to predict classes
    lines = content.splitlines()
    paragraph = ''
    attachm_paragraph = ''

    for line in lines:
        # If new paragraph in file
        if line.strip():
            line = re.sub('\W+',' ', line)
            paragraph = paragraph + ' ' + line
        else:
            # Get predicted class
            result = svc_model.predict(tfidf.transform([paragraph]))
            result = classes[result[0]]
            # Get confidence probability
            prob = calibrated_svc.predict_proba(tfidf.transform([paragraph]))
            prob = max(prob[0])

            # # Write every detection and probability to txt
            # with open(model_path + "Detected/" + agreement_type + "_all_probs_ " + agreement + ".txt", "a+") as f:
            #     f.write(result + " " + str(prob) + "\n" + paragraph + "\n")

            if prob > 0.13: # Prediction threshold
                for i in range(0, len(classes)):
                    if result == classes[i]:
                        sections[i] += paragraph
                # # Write every detection and probability to txt
                # with open(model_path + "Detected/" + agreement_type + "_db_probs_ " + agreement + ".txt", "a+") as f:
                #     f.write(result + " " + str(prob) + "\n" + paragraph + "\n")
            else:
                attachm_paragraph += paragraph
            paragraph = ''

    return sections, attachm_paragraph


def InsertIntoDB(name, classes, sections):
    db = pymysql.connect('localhost', 'ealloem', 'Ericsson1', 'ai')
    cursor = db. cursor()
    try:
        classes = list(classes.values())
        # Columns
        sql = "INSERT INTO " + agreement_type + " ("
        classes_name = "Agreement_name, "
        for i in range(len(classes)-1):
            classes_name = classes_name + classes[i] +", "
        classes_name = classes_name + classes[len(classes)-1]

        sql = sql + classes_name + ") VALUES("
        # Rows
        name = name.replace('.txt', '.pdf').replace("'", "\'")
        sections_content = "\""+name+"\""+", "
        for i in range(len(sections)-1):
            sections_content = sections_content + "\'"+ sections[i] + "\'" + ", "
        sections_content = sections_content + "\'"+ sections[len(sections)-1] + "\'" 

        sql = sql + sections_content + ");"

        # # Write query to txt
        # with open(model_path + agreement_type + "_classified.txt", "a+") as f:
        #     f.write(sql + "\n")

        cursor.execute(sql)
        db.autocommit(True)
        print('-- Agreement successfully added to the DB. --\n')
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print("Exeception occured  line {} : {}".format(exc_tb.tb_lineno,e))
        # Write failed queries to txt
        with open(model_path + agreement_type + "_failed_to_db.txt", "a+") as f:
            f.write(name + "\n")

# # # # # M A I N  F U N C T I O N # # # # #

if __name__ == "__main__":
    folder = glob.glob(new_data_path+"*.txt")
    if not folder:
        print("There are no files to scan in the folder.")
        sys.exit()

    folder.sort()
    print("{} agreements found".format(len(folder)))
    i = 0
    classes, model, calibrated_svc, tfidf = Train_Model()   # Train model

    for file in folder:
        i += 1
        name = file
        file = os.path.basename(file)                                                   # gets file name from path
        print("\n- Extracting sections: {}".format(os.path.splitext(file)[0]))          # prints file name without extension .pdf

        sections, attachm_paragraph = Predict(file, classes, model, calibrated_svc, tfidf) # Get sections detected
        print(i)

        try:
            InsertIntoDB(file, classes, sections)
        except Exception as e:
            print("Exeception occured: {}".format(e))