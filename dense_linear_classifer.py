import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")
#print(df)
# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]
data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}
# convert a document into a vector
TB_tokenizer =  TreebankWordTokenizer()

def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    # TODO: tokenize the input document
    toks = TB_tokenizer.tokenize(doc)
    # TODO: aggregate the vectors of words in the input document
    text_vecs = [w2v[word] for word in toks if word in w2v]
    # calculate the mean of the vectors and return
    if len(text_vecs) > 1:
        res =  np.mean(text_vecs, axis = 0)
        #print(len(res))
        return res
    elif len(text_vecs) == 1:
        return res[0]
    else:
        return np.zeros((1,300))
    
    #return vec
            
# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    #TODO: convert each of the training documents into a vector
    X= [document_to_vector(x) for x in Xtr]

    #TODO: train the logistic regression classifier
    lr = LogisticRegression(C=C)
    lr.fit(X, Ytr)
    
    
    return lr  
   

# fit a linear model 
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    
    #TODO: convert each of the testing documents into a vector
    Xtst= [document_to_vector(x) for x in Xtst]
         # X= list of review vectors

    #TODO: test the logistic regression classifier and calculate the accuracy
    y_pred = model.predict(Xtst)
    score = round(accuracy_score(Ytst,y_pred),3)
    print("Accuracy: ",score)
    return score


# TODO: search for the best C parameter using the validation set
#finding best C
c_values = [0.1, 1.0, 5.0, 20.0, 50.0, 60.0, 100.0, 200.0, 1000.0]
X= [document_to_vector(x) for x in Xr]
for c in c_values: 
    lr = LogisticRegression(C=c, solver = 'liblinear')
    scores = cross_val_score(lr, X, Yr, cv=5, scoring = 'accuracy')     
    print("C = ", c, ": ", round(scores.mean(), 5))

# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result
# test sets
model = fit_model(Xr, Yr, 20)
test_model(model, X_test, Y_test)

#test + validation sets
model = fit_model(Xr + X_val, Yr + Y_val, 20)
test_model(model, X_test, Y_test)