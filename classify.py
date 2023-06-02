import extract_features
import numpy as np
import pickle 
import matplotlib.pyplot as plt


def classify(img, mask):
   '''This function takes a picture of a leison and its corresponding mask, both opened as plt.imread, and predicts
   if its healthy (1) or not (0).'''
    
   #Extract features
   X = extract_features(img, mask)

   #Convert X to a NumPy array and reshape it
   X = np.array(X).reshape(1,-1)
    
   #Load the trained classifier
   classifier = pickle.load(open('group8_classifier.sav', 'rb'))
    
   #Use it on this example to predict the label AND posterior probability
   pred_label = classifier.predict(X)
   pred_prob = classifier.predict_proba(X)
     
   print('predicted label is ', pred_label)
   print('predicted probability is ', pred_prob)

   return pred_label, pred_prob

def main():

    img = plt.imread('PAT_55_84_506.png')
    mask = plt.imread('mask_PAT_55_84_506.png')

    classify(img,mask)


if __name__ == '__main__':
    main()