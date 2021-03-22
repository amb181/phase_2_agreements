import re, pymysql, glob, os,sys, pytesseract, inflect, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pickle
import sys
from cv2 import cv2
from pdf2image import convert_from_path
from PyPDF2 import PdfFileWriter, PdfFileReader
# # # # # Folder path to test MPAs # # # # #

model_info_path = "/home/rasa-stakeholder-assistant/agreements/phase2/MPA/MPA_model_info/"
path_test = "/home/rasa-stakeholder-assistant/agreements/phase2/MPA/test/"
#path_test = "/home/rasa-stakeholder-assistant/agreements/phase2/temp/"
start_flag = True



def train_mpa_model():
    # # # # # T R A I N I N G # # # # #

    # Read csv file
    df = pd.read_csv("/home/rasa-stakeholder-assistant/agreements/phase2/MPA/MPA_Training.csv",encoding = "ISO-8859-1")
    df.head()

    # Get columns to work on
    col = ['Section', 'Paragraph']
    df = df[col]
    df = df[pd.notnull(df['Paragraph'])]
    df.columns = ['Section', 'Paragraph']
    df['Category_ID'] = df['Section'].factorize()[0]
    category_id_df = df[['Section', 'Category_ID']].drop_duplicates().sort_values('Category_ID')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['Category_ID', 'Section']].values)
    df.head()
    print (id_to_category)

    # # Plot how many examples per section
    # fig = plt.figure(figsize=(8,6))
    # df.groupby('Section').Paragraph.count().plot.bar(ylim=0)
    # plt.show()

    #Vectorize Data
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')
    features = tfidf.fit_transform(df['Paragraph'].values.astype('U')).toarray()
    labels = df['Category_ID']
    features.shape

    # # Find most correlated terms
    # N = 2
    # for Section, Category_ID in sorted(category_to_id.items()):
    #   features_chi2 = chi2(features, labels == Category_ID)
    #   indices = np.argsort(features_chi2[0])
    #   feature_names = np.array(tfidf.get_feature_names())[indices]
    #   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    #   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    #   print("# '{}':".format(Section))
    #   print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    #   print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

    # Train model
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0,stratify =labels )
    model.fit(X_train, y_train)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    y_pred = model.predict(X_test)

    # Wrap model to get probabilities in prediction
    calibrated_svc = CalibratedClassifierCV(model, method='sigmoid')
    calibrated_svc.fit(X_train, y_train)

    # # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred,normalize = 'true')
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='.2', xticklabels=category_id_df.Section.values, yticklabels=category_id_df.Section.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show() 
 
    # # Print report for each class
    print(metrics.classification_report(y_test, y_pred, target_names=df['Section'].unique()))
    # save the model and variables to disk
    pickle.dump(id_to_category, open(os.path.join(model_info_path+"classes.sav"), 'wb'))
    pickle.dump(model, open(os.path.join(model_info_path+"MPA_model.sav"), 'wb'))
    pickle.dump(calibrated_svc, open(os.path.join(model_info_path+"calibrated_svc.sav"), 'wb'))
    pickle.dump(tfidf, open(os.path.join(model_info_path+"vectorizer_fit.sav"), 'wb'))

    return id_to_category, model, calibrated_svc, tfidf

def predict_MPA(agreement, classes, model, calibrated_svc,tfidf):
    # # # # # P R E D I C T I O N # # # # #
    
    send_to_att = False
    #ind =list(classes.values()).index('Attachment')
    exhibits_parag = ""
    #print(ind)
    print(path_test+agreement)
    with open(path_test+agreement) as f:
        content = f.read()
    #print(content)
    # Sections to save into DB
    sections = [""] * len(classes)

    # Read agreement line by line and build paragraphs to predict classes
    lines = content.splitlines()
    paragraph = ''
    for line in lines:

        if line.strip():
            line = re.sub('\W+',' ', line)
            paragraph = paragraph + " " + line
            if "EXHIBIT A" in line:
                send_to_att = True
        else:
            # Get predicted class
            if not send_to_att:
                result = model.predict(tfidf.transform([paragraph]))
                result = classes[result[0]]
                # Get confidence probability
                prob = calibrated_svc.predict_proba(tfidf.transform([paragraph]))
                prob = max(prob[0])
                if prob > 0.13: # Prediction threshold
                    for i in range(0,len(classes)):
                        if result == classes[i]:
                            sections[i] += paragraph
            else:

                exhibits_parag += paragraph
            paragraph = ''
    return sections, exhibits_parag


def insert_into_DB(agreement, parties, scope, conf_info, employee, affiliates, duration, duty, exc_info, rights, compliance, notice, \
            assignment, amendments, ctrl_law, expiration, injunctive, attorneys, counterparts, miscellaneous):
    # Insert into DB
    db = pymysql.connect('localhost', 'efgaorm', 'Ericsson1', 'ai')
    try:
        cursor = db.cursor()
        sql = "INSERT INTO MPA (id, Agreement_name, Summary , Recitals_And_Intent , Definitions , General_Scope_Of_Agreement , \
        Term_Of_Agreement , Suppliers_Requirements , Forecast_Procedure_And_Product_Availability , Prices , Ordering_Procedure , \
        Product_Shipping_And_Delivery , Delays , Acceptance_Invoicing_And_Payment , Continuity_Of_Supply_And_Product_Substitution , Documents_And_Production_Equipment , Expiration_of_Agreement, \
        Quality_And_Inspection , Engineering_Change_Request , Warranties, Training, Indemnification , Limitation_Of_Liability , \
        Insurance, Termination, Force_Majeure , Compliance ,Confidentiality ,M_W_Dvbe_Participation_Plans_And_Reports ,Miscellaneous \
        Product_Claims ,Supplier_S_Marketing_Demonstration_And_Testing_Obligations,Ownership_Of_Intellectual_Property_Rights ,\
        New_Products_And_Technical_Modifications , Diverse_Supplier_Participation_Plans_And_Reports, Dispute_Resolution_Governing_Law,\
        Notices , Liens , Publicity_Communication_And_Trademarks, Relationship_Of_Parties, Location_Of_Services , Assignment_Amendments_Survival ) \
        VALUES (null,'%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s');"\
        % (agreement, parties, scope, conf_info, employee, affiliates, duration, duty, exc_info, rights, compliance, notice, \
            assignment, amendments, ctrl_law, expiration, injunctive, attorneys, counterparts, miscellaneous)
        # print(sql)
        cursor.execute(sql)
        db.autocommit(True)
        db.close()
        print('\n-- MPA agreement successfully added to the DB. --\n')
    except Exception as e:
        print("Exception occured: {}".format(e))

    # print(parties, scope, conf_info, employee, affiliates, duration, duty, exc_info, rights, compliance, notice, \
    #          assignment, amendments, ctrl_law, expiration, injunctive, attorneys, counterparts, miscellaneous, sep="\n")



def InsertIntoDB(name,classes, sections,exhibits):
    #db = pymysql.connect('localhost', 'efgaorm', 'Ericsson1', 'ai')
    sql_file = "contracts_info_sql.txt"
    if start_flag:
        if os.path.exists(sql_file):
            os.remove(sql_file)
    try:
        classes = list(classes.values())
        sql = "INSERT INTO MPA ("
        classes_name = "Agreement_name,"
        for i in range(len(classes)-1):
            classes_name = classes_name + classes[i] +", "
        classes_name = classes_name + classes[len(classes)-1]

        sql = sql + classes_name + ",Exhibit"+") VALUES("
        
        sections_content = "'"+name+"'"+","
        for i in range(len(sections)-1):
            sections_content = sections_content + "'"+ sections[i] + "'" + ", "
        sections_content = sections_content + "'"+ sections[len(sections)-1] + "'" + ", '"+ exhibits + "'"

        sql = sql + sections_content + ");"
        with open(sql_file, "a+") as f:
            f.write(sql + "\n")

        #cursor.execute(sql)
        #db.autocommit(True)
        #print('\n-- Fields successfully added to the DB. --\n')
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print("Exeception occured  line {} : {}".format(exc_tb.tb_lineno,e))



def PDF2Img(name):
    print('Transforming PDF into images...')
    """ Split .pdf file into single pages """
    inputpdf = PdfFileReader(open(name, "rb"), warndest=None, strict = False, overwriteWarnings=True)
    name = os.path.splitext(name)[0]                        # Get filename without extension .pdf

    for i in range(inputpdf.numPages):
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        if i < 10:
            page = name+'_0'+str(i)
        else:
            page = name+'_'+str(i)

        pdf_page = page + ".pdf"
        with open(pdf_page, "wb") as outPutStream:
            output.write(outPutStream)

        """ Transform to image """
        img = convert_from_path(pdf_page, fmt='jpeg', dpi=600)
        for image in img:
            image.save(page + '.jpg')
        os.remove(pdf_page)

def ConcatenateImgs(name):
    print('Getting text from images...')
    folder = glob.glob(path_test+"*.jpg")
    folder.sort()
    i = 0 
    if os.path.exists(name):
        os.remove(name)
    for file in folder:
        img = cv2.imread(file, 0)

        """Convert image to text"""
        text = pytesseract.image_to_string(img, lang='engfast') # engfast, new_eng_best
        if i == 0:
            name = os.path.splitext(name)[0]
        txt_file = open(name+".txt", "a+")
        txt_file.write(text)
        txt_file.write("\n-- END OF PAGE {} --\n".format(i+1))
        txt_file.flush()
        txt_file.close()
        i += 1
        os.remove(file)                                                     # Delete image

        # """ Save img in .tif format to train model """
        # new_img = os.path.splitext(file)[0]
        # cv2.imwrite('{}.tif'.format(new_img), image_p)

    print("{} pages written in {}".format(i, os.path.basename(txt_file.name))) 
    return i


#process = "Digitalize"
process = "Classify"

train_process =False
# # # # # M A I N  F U N C T I O N # # # # #

if __name__ == "__main__":
    ext="pdf"
    if process is "Classify":
        ext="txt"

    folder = glob.glob(path_test+"*."+ext)
    if not folder:
        print("There are no files to scan in the folder.")
    folder.sort()
    print("{} agreements found".format(len(folder)))
    i = 0
    if train_process:
        classes, model, calibrated_svc,tfidf = train_mpa_model()   # Train model
    else:
        # load the model from disk
        classes = pickle.load(open(os.path.join(model_info_path+"classes.sav"), 'rb'))
        model = pickle.load(open(model_info_path+"MPA_model.sav", 'rb'))
        calibrated_svc = pickle.load(open(model_info_path+"calibrated_svc.sav", 'rb'))
        tfidf = pickle.load(open(model_info_path+"vectorizer_fit.sav", 'rb'))

    numbr = 1
    for file in folder:
        if True: #numbr in range (20,40):
            i += 1
            name = file
            file = os.path.basename(file)                                        # gets file name from path
            print("\n- Processing: {}".format(os.path.splitext(file)[0]))         # prints file name without extension .pdf
            # Detection
            if process is "Digitalize":
                PDF2Img(name)
                pages = ConcatenateImgs(name)
            if process is "Classify":
                sections, exhibits_parag = predict_MPA(file, classes, model, calibrated_svc,tfidf) # Get sections detected
                #Save detected sections into a txt file
                # with open(path_test + os.path.splitext(file)[0] + "_sectioned.txt", "a+") as f:
                #     for i in range(0,len(classes)):
                #         f.write(classes[i] + "\n" + sections[i] + "\n****************************************************************************************\n")
                #     f.write("EXHIBITS" + "\n" + exhibits_parag + "\n*******************************************************************************************\n")
            
                #Insert into DB
                pdf_name = os.path.splitext(file)[0] + ".pdf"
                print(pdf_name)
                try:
                    InsertIntoDB(pdf_name,classes,sections,exhibits_parag)
                except Exception as e:
                    _, _, exc_tb = sys.exc_info()
                    print("Exeception occured  line {} : {}".format(exc_tb.tb_lineno,e))
            
            start_flag = False
        numbr = numbr + 1