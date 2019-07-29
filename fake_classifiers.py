import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


df = shuffle(pd.read_csv('news_dataset.csv').dropna())
# Fake = 1
df['label'] = np.where(df['label']=='fake',1,0)
n = 4000
print('Data size', n)
df = df[:n]
# print(df.columns)
# print(df[['content','title']].head())
X_train, X_test, y_train, y_test = train_test_split(df[['content', 'title']], 
                                                    df['label'], 
                                                    random_state=0)
                            

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

class SentLenExtractor(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
      # self.vars = vars  # e.g. pass in a column name to extract
  def average_sent_len(self, x):
    if len(x) == 0:
      return 0
    sents = sent_tokenize(x)
    if len(sents) == 0:
      return 1
    # print(sents)
    # print(sents[0])
    lens = [len(word_tokenize(sent)) for sent in sents]
    # print(lens)
    return sum(lens) / len(lens)

  def transform(self, X, y=None):
    return pd.DataFrame(pd.Series(X['content']).apply(self.average_sent_len)).fillna(0)  # where the actual feature extraction happens

  def fit(self, X, y=None):
    return self  # generally does nothing

# def naive_bayes():
#     vect = TfidfVectorizer(min_df=3).fit(X_train)
#     clfrNB = MultinomialNB(alpha=0.1)
#     clfrNB.fit(vect.fit_transform(X_train), y_train)
#     predictions = clfrNB.predict(vect.transform(X_test))    
#     return roc_auc_score(y_test, predictions)

# def add_sentence_length_feature_to_df(X):
#   X['sent_lens'] = X['content'].apply(average_sent_len)
#   return X

def get_title(x):
  # print(x['title'])
  return x['title']

def get_content(x):
  # print('cont', type(x['content']))
  return x['content']

def classifier_pipe():
  stops = stopwords.words('english')
  title_pipe = Pipeline([('extract_field',
                              FunctionTransformer(get_title,
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(ngram_range=(1,3), lowercase=False, stop_words=stops))])
  content_pipe =  Pipeline([('extract_field',
                              FunctionTransformer(get_content,
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(min_df=3, ngram_range=(1,9), lowercase=False, stop_words=stops))])
  sent_len_pipe = Pipeline([('extract_field', SentLenExtractor())])


  transformer = [('FeatureUnion',FeatureUnion([
                  ('Avg_Sent_Len', SentLenExtractor()),
                  ('title_tfidf', title_pipe),
                  ('content_tfidf', content_pipe)
                              ]))]

  transformer.append(('clf',LogisticRegression(solver='sag', n_jobs=-1, C=100, max_iter=8000)))                    
  return Pipeline(transformer)



def general_gridsearch():
  print(type(X_train['content']))
  pipe = classifier_pipe()

  # print(pipe.get_params().keys())
  # param_grid = dict(clf__C=[1,100,1000])
  param_grid = dict()
  # param_grid = dict(FeatureUnion__content_tfidf__tfidf__ngram_range=[(1,9)])
  # param_grid = dict(clf__solver=['liblinear','sag','newton-cg','lbfgs'])
                      # FeatureUnion__content_tfidf__tfidf__lowercase=[True,False],
  model = GridSearchCV(pipe, param_grid=param_grid, cv=2)
  model.fit(X_train, y_train)


  print(pipe)
  print("score = %3.2f" %(model.score(X_test, y_test)))
  print(model.best_params_)

  y_score = model.decision_function(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, y_score)
  roc_auc = roc_auc_score(y_test, y_score)
  predictions = model.predict(X_test)
  print('roc_auc:', roc_auc)
  print('precision:',precision_score(y_test, predictions))
  print('recall:',recall_score(y_test, predictions))
  # plt.figure(figsize=(16, 12))
  # plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
  # plt.plot([0, 1], [0, 1], 'k--')
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  # plt.xlabel('False Positive Rate (1 - Specificity)', size=16)
  # plt.ylabel('True Positive Rate (Sensitivity)', size=16)
  # plt.title('ROC Curve', size=20)
  # plt.legend(fontsize=14)
  # plt.show()

# print('log with pipe roc, precision, recall:', log_reg_with_pipe())
# gridsearch_over_log()
general_gridsearch()


"""
TODO
refine list of stopwords
understand what a roc list is
map all words to a pos tag
then run tfidf/countvectorizer on ngrams of mapping
maybe try sentence trees?
sentence lengths?
"""

# Precision oriented minimizes false positives -- labeling real as fake
# Recall oriented minimizes false negatives-- labeling fake as real

"""
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
 "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
 "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
  'won', "won't", 'wouldn', "wouldn't"]
"""
