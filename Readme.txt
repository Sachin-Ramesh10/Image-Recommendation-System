# Final testing is dependent on the dataset, without the data program would throw an error
# all python notebooks are saved as html for easy viewing


Crawling:
 
> user_queue.txt -- has user list to extract data from
> User_crawl.py  --program to extract board information from which images will be downloaded from
> SaveImagesCrawl.py -- program to download and save images(creating original dataset)
> boards.csv -- file will be accessed to download images as it contains user urls and selected boards.

Data Resructuring:

> classes.csv -- This file contains the information about the created classes and the boards that are assigned to each classes
> restructure.ipynb -- this file refers the classes.csv and copies the images from original dataset arrranging into referred classes
> Final restructure.csv -- deprecates the skewed images from the dataset making it uniform numer of images for training

Image Classification:

> cnn.py -- training the dataset with vanilla CNN network
> Inception.py -- trains the dataset with Inception network with imagenet weights
> _InceptionContinue.py -- loads the trained inception model from before for further training using finetuning
> Xception.py -- trains the dataset with Xception network with imagenet weights
> VGG16.py -- trains the dataset with VGG16 network with imagenet weights
> VGG16_continue -- loads the trained VGG16 model from before for further training using finetuning

Results:

> Final_Prediction.ipynb -- this python notebook shows the partial results(without displying images) but helps understand the code straucture better
> Final_Prediction.py -- This is the final prediction program ( executed in the format python Final_Prediction.py imageundertest)
> Final_results.docx -- Complete output of the prediction
> inception pred.ipynb -- Shows the result of test accuracy of inception model
> Xceptionpreds.ipynb -- shows the result of test accuracy of Xception model

TextData Creation:

> Description.csv -- Created dataset for description based text classification
> urls.csv -- urllink for the sentences to be downloaded from the website
> Sentence Creation.ipynb -- Creates the text corpus by downloading the sentences from urls mention inthe urls files
> text.csv -- corpus created for text classification

Text Training:

> Text_train.ipynb -- Trains the corpus text.csv
> description.ipynb -- Trains the description.csv file
> CNN_LSTM.ipynb -- Shows the result of CNN and LSTM models for text.csv corpus

Training screenshots:
> Has the training screenshots of image classification training.