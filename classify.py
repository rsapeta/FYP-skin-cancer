from extract_features import extract_features
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.color import rgba2rgb
from skimage.segmentation import slic


def classify(img, mask):
   '''This function takes a picture of a leison and its corresponding mask, both opened as plt.imread, and predicts
   if its healthy (1) or not (0).'''
    
   #Extract features
   X = extract_features(img, mask)
   features = ["Asymmetry", "Color Variability (R)", "Color Variability (G)", "Color Variability (B)",
                           "Compactness", "Average Color (R)", "Average Color (G)", "Average Color (B)"]
   
   df = pd.DataFrame([X],columns= features)
   X=df
    
   #Load the trained classifier
   classifier = pickle.load(open('group8_classifier.sav', 'rb'))
    
   #Use it on this example to predict the label AND posterior probability
   pred_label = classifier.predict(X)
   pred_prob = classifier.predict_proba(X)
     
   print('predicted label is ', pred_label)
   print('predicted probability is ', pred_prob)

   return pred_label, pred_prob

def main():

    #img = plt.imread('')
    #mask = plt.imread('')

    classify(img,mask)


if __name__ == '__main__':
    main()