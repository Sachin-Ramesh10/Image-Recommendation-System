import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import pandas as pd
from keras.preprocessing.image import load_img



def respred(input_):
    print("")
    print("Now Loading resnet50 for prediction")
    print("")
    resmodel = ResNet50(weights='imagenet')
    image = load_img(input_, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = resmodel.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    class_,percentage = (label[1], label[2]*100)
    print("")
    print("Loaded Resnet50 predicted the input image as {} with {}% confidence".format(class_,round(percentage,2)))
    print("")

from glob import glob
datapaths = glob('Data/Train/*/*')
ub = []
for i in datapaths:
    a = i.split('/')[-1]
    ub.append((i,a))

userboards = pd.DataFrame(ub)
userboards.columns=(['paths','board'])

test_path = glob('/Users/arjunkm/Desktop/sachin1/proj/project/Test/*')
classes= []
for i in test_path:
    if i not in classes:
        classes.append(i.split("/")[-1])
        
headers = ["class","boards"]
catboard = pd.read_csv('Classes.csv', names= headers)


def display_users(classname):

    ww = "https://www.pinterest.com/"
    brd = catboard.loc[catboard['class'] == classname]

    bnames = brd['boards'].tolist()[0].split(',')

    userpaths = []
    for i in bnames:
        paths = userboards.loc[userboards['board'] == i]
        paths = paths['paths'].tolist()[0]
        userpaths.append(paths)

    print("")
    print("Now displaying usernames and boards with similar images")
    print("")
    print("======================================================")
    print("Username\t\t\t\tBoards")
    print("======================================================")
    for i in userpaths:
        uname = i.split('/')[-2]
        bname = i.split('/')[-1].replace(" ","")
        print(str(uname)+"\t\t\t\t"+(bname))
        print("Pinterest Link {}\n".format(ww+uname+"/"+bname))
    print("======================================================")
    print("")
    print("Do you Want to view the Sample images from above userboard(s)?(yes/no)")
    print("")
    text = input("") 

    if text == "yes":
        import os, random
        import cv2                
        import matplotlib.pyplot as plt   
        for i in userpaths:
            file = i+"/"+random.choice(os.listdir(i))
            img = cv2.imread(file)
            title = i.split("/")[-2]+"/"+i.split("/")[-1]
            plt.title(title)
            plt.imshow(img)
            plt.show()


def inceppred(ims):
    
    from keras.models import load_model
    from keras.applications.inception_v3 import preprocess_input
    from PIL import Image
    target_size = (299, 299)
    import numpy as np

    print("")
    print("Now Loading inceptionv3 for prediction")
    print("")
    insmodel = load_model('Models/Inception/inceptionv3-ft.model')
    
    img = Image.open(ims)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = insmodel.predict(x)[0]
    y_classes = np.argmax(preds)

    prediction = ""
    if(int(y_classes) > 30):
        prediction = classes[y_classes+1] 
    else:
        prediction = classes[y_classes]

    print("")
    print("Fine Tuned Inception model predicted the input image as {} with {}% confidence".format(prediction,round(max(preds)*100,2)))
    print("")
    display_users(prediction)


import unicodedata
import sys
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


def remove_punctuation(text):
    return text.translate(tbl)

def text_prediction(insent,flag):


    import numpy as np
    import tflearn
    import tensorflow as tf
    import random
    import json
    import string
    import unicodedata
    
    global words
    
# variable to hold the Json data read from the file
    data = None

    # read the json file and load the training data
    if flag ==1:
        with open('dat.json') as json_data:
            data = json.load(json_data)
    else:
        if flag == 2:
            with open('desc.json') as json_data:
                data = json.load(json_data)
    categories = list(data.keys())
    words = []
# a list of tuples with words in the sentence and category name
    docs = []
    for each_category in data.keys():
        for each_sentence in data[each_category]:
            # remove any punctuation from the sentence
            each_sentence = remove_punctuation(each_sentence)
            #print(each_sentence)
            # extract words from each sentence and append to the word list
            w = nltk.word_tokenize(each_sentence)
            #print("tokenized words: ", w)
            words.extend(w)
            docs.append((w, each_category))
# stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))
    training = []
    output = []
# create an empty array for our output
    output_empty = [0] * len(categories)
    for doc in docs:
    # initialize our bag of words(bow) for each document in the list
        bow = []
        # list of tokenized words for the pattern
        token_words = doc[0]
        # stem each word
        token_words = [stemmer.stem(word.lower()) for word in token_words]
        # create our bag of words array
        for w in words:
            bow.append(1) if w in token_words else bow.append(0)

        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1

        # our training set will contain a the bag of words model and the output row that tells
        # which catefory that bow belongs to.
        training.append([bow, output_row])
# shuffle our features and turn into np.array as tensorflow  takes in numpy array
    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

# reset underlying graph data
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)
    modeltext = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    if flag ==1:
        print("\n Now Loading model for text classification\n")
        modeltext.load('model.tflearn')
    else:
        if flag ==2:
            print("\n Now Loading model for decription classification\n")
            modeltext.load('descmodel.tflearn')

    # tokenize the pattern
    sentence_words = nltk.word_tokenize(insent)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    p = modeltext.predict([np.array(bow)])
    return categories, p


def sentence(flag):
    import numpy as np
    senten = ""
    if flag ==1:
        print("")
        print("Enter your text for classification")
        print("")
        senten = input("")
    else:
        if flag == 2:
            print("")
            print("Enter your description for classification")
            print("")
            senten = input("")
    
    categories,  pre = text_prediction(senten, flag)
    if flag == 1:
        print("")
        final = categories[np.argmax(pre)]
        print("Text Classification Model predicted the input Sentence as {} with {}% confidence\n".format(final,round(np.max(pre)*100,2)))
        display_users(final)
    else:
        if flag == 2:
            print("")
            print("The description has been classified among below class(es)")
            count = 0
            final = pre[0].tolist()
            clist = []
            for i in final:
                if i >= (max(final)-0.2) and i <= (max(final)+0.2):
                    print(categories[final.index(i)],round(i*100,2))
                    clist.append(categories[final.index(i)])
                    count+=1
            if count>1:
                print("\nDo you want to enter more specific description?(yes/no)\n")
                descri = input("")
                if descri == "yes":
                    sentence(2)
            try:
                
                sa = categories[final.index(max(final))]
                display_users(sa)
                exit(0)
            except Exception:
                exit(0)
                            


import sys
def main(argv):

    if len(sys.argv) > 2:
        
        if argv[1]=="resnet":
            respred(argv[0])
            exit(0)
        if argv[1]=="Inception":
            inceppred(argv[0])
            exit(0)
        if argv[1]=="Text":
            sentence(1)
            exit(0)
        if argv[1]=="Description":
            sentence(2)
            exit(0)
    else:
        
        respred(argv[0])
        inceppred(argv[0])
        sentence(1)
        sentence(2)


            
if __name__ == "__main__":
    main(sys.argv[1:])
