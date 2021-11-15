# NLP-text-classifiers
Text classification tasks: sentiment classifier and genre classifier
(data not uploaded due to file size)
- dense_linear_classifier: A IMDB movie review sentiment classifier (positive/negative) trained with Logistic Regression
  Train accuracy: ~ 85.23
  Test accuracy: ~ 85.5
  
- genre_classifier_0pc: A document(novel) genre classifier RNN model fine-tuned with pretrained-BERT, predicting the text peices into horror(class id 0), 
science fiction (class id 1), humor (class id 2), and crime fiction(class id 4)
  train F1 score: ~ 0.89
  test F1 score: ~ 0.952
