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
import matplotlib.pyplot as plt


df = shuffle(pd.read_csv('news_dataset.csv').dropna())
# Fake = 1
df['label'] = np.where(df['label']=='fake',1,0)
n = 1000
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


# def naive_bayes():
#     vect = TfidfVectorizer(min_df=3).fit(X_train)
#     clfrNB = MultinomialNB(alpha=0.1)
#     clfrNB.fit(vect.fit_transform(X_train), y_train)
#     predictions = clfrNB.predict(vect.transform(X_test))    
#     return roc_auc_score(y_test, predictions)


# def support_vec_just_content():
#     vect = TfidfVectorizer(min_df=5)
#     svm = SVC(C=0.1, kernel='linear')
#     X_train_vectorized = vect.fit_transform(X_train)
#     # print(X_train_vectorized.shape)
#     svm.fit(X_train_vectorized, y_train)
#     X_test_vectorized = vect.transform(X_test)
#     predictions = svm.predict(X_test_vectorized)
#     # print(predictions)
#     # print(y_test)
#     return roc_auc_score(y_test, predictions, average='macro')   



def title_content_pipe():
    transformer = [('FeatureUnion',FeatureUnion([
                ('title_tfidf',
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['title'],
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(ngram_range=[3,5]))])),
                ('content_tfidf',
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['content'],
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(min_df=3, ngram_range=(2,4)))])),]))]

    transformer.append(('svm', SVC(C=0.1, kernel='linear')))                    
  
    return Pipeline(transformer)

# def support_vec_with_lens():
#     """Returns auc, precision, recall."""
#     vect = TfidfVectorizer(min_df=3, ngram_range=[3,5])
#     title_vect = TfidfVectorizer(ngram_range=(2,4)) 
#     svm = SVC(C=0.1, kernel='linear')

    
#     X_train_vectorized = vect.fit_transform(X_train['content'])
#     titles_vectorized = title_vect.fit_transform(X_train['title'])
#     txt_lens = X_train['content'].str.len()
#     X_train_vectorized = add_feature(titles_vectorized, titles_vectorized)
#     X_train_vectorized = add_feature(X_train_vectorized, txt_lens)
#     svm.fit(X_train_vectorized, y_train)


#     X_test_vectorized = vect.transform(X_test['content'])
#     test_titles_vectorized = title_vect.transform(X_test['title'])
#     test_txt_lens = X_test['content'].str.len()
#     X_test_vectorized = add_feature(X_test_vectorized, test_titles_vectorized)
#     X_test_vectorized = add_feature(X_test_vectorized, test_txt_lens)

#     predictions = svm.predict(X_test_vectorized)
#     return (roc_auc_score(y_test, predictions, average='micro'),
#             precision_score(y_test, predictions),
#             recall_score(y_test, predictions))

# def support_vec_with_pipe():
#     """Returns auc, precision, recall."""
#     pipe = title_content_pipe()
#     pipe.fit(X_train, y_train)
#     predictions = pipe.predict(X_test)
#     return (roc_auc_score(y_test, predictions, average='micro'),
#             precision_score(y_test, predictions),
#             recall_score(y_test, predictions))

# def log_reg_with_pipe():
#     """Returns auc, precision, recall."""
#     pipe = log_pipe()
#     pipe.fit(X_train, y_train)
#     predictions = pipe.predict(X_test)
#     return (roc_auc_score(y_test, predictions, average='micro'),
#             precision_score(y_test, predictions),
#             recall_score(y_test, predictions))
# def log_reg_model():
#     # import re
#     # def digit_counter(text):
#     #     count = 0
#     #     for w in text:
#     #         if w.isdigit():
#     #             count += 1
#     #     return count
    
#     # def count_non_word(text):
#     #     return len(re.findall(r'\W', text))
    
#     vect = CountVectorizer(min_df=5, ngram_range=(3,5), analyzer='char_wb')
#     length_of_doc = X_train.str.len()
#     # digit_count = X_train.apply(digit_counter)
#     # non_word_char_count = X_train.apply(count_non_word)
    
    
#     X_train_vectorized = vect.fit_transform(X_train)
#     # X_train_vectorized = add_feature(X_train_vectorized, length_of_doc)
#     # X_train_vectorized = add_feature(X_train_vectorized, digit_count)
#     # X_train_vectorized = add_feature(X_train_vectorized, non_word_char_count)
    
#     model = LogisticRegression(C=100)
#     model.fit(X_train_vectorized, y_train)
    
#     X_test_vectorized = vect.transform(X_test)
#     length_of_doc = X_test.str.len()
#     # digit_count = X_test.apply(digit_counter)
#     # non_word_char_count = X_test.apply(count_non_word)
#     # X_test_vectorized = add_feature(X_test_vectorized, length_of_doc)
#     # X_test_vectorized = add_feature(X_test_vectorized, digit_count)
#     # X_test_vectorized = add_feature(X_test_vectorized, non_word_char_count)
    
#     predictions = model.predict(X_test_vectorized)
    
#     sorted_coef_index = model.coef_[0].argsort()
#     feature_names = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
# #     print(len(model.coef_[0].argsort()), len(feature_names))
#     return (roc_auc_score(y_test, predictions),
#             pd.Series(feature_names[sorted_coef_index[:10]]),
#             pd.Series(feature_names[sorted_coef_index[:-11:-1]]))

# print('naive_bayes:', naive_bayes())
# print('SVM simple', support_vec_just_content())
# print('SVM with lens roc, precision, recall:', support_vec_with_lens())

# print('SVM with pipe roc, precision, recall:', support_vec_with_pipe())

def log_pipe():
    transformer = [('FeatureUnion',FeatureUnion([
                ('title_tfidf',
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['title'],
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(ngram_range=(2,4)))])),
                ('content_tfidf',
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['content'],
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(min_df=3, ngram_range=(3,5)))])),]))]

    transformer.append(('log_reg',LogisticRegression(solver='liblinear')))                    
  
    return Pipeline(transformer)



def gridsearch_over_log():
  pipe = log_pipe()
  param_grid = dict(FeatureUnion__content_tfidf__tfidf__min_df=[1, 2, 3, 5], log_reg__C=[10, 100, 1000])
  grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=2)
  grid_search.fit(X_train, y_train)
  print("score = %3.2f" %(grid_search.score(X_test, y_test)))
  print(grid_search.best_params_)

def classifier_pipe():
  stops = stopwords.words('english')
  transformer = [('FeatureUnion',FeatureUnion([
                ('title_tfidf',
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['title'],
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(ngram_range=(1,3), lowercase=False, stop_words=stops))])),
                ('content_tfidf',
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['content'],
                                                  validate=False)),
                            ('tfidf',
                              TfidfVectorizer(min_df=3, ngram_range=(1,9), lowercase=False, stop_words=stops))])),
                              ]))]

  transformer.append(('clf',LogisticRegression(solver='sag', n_jobs=-1, C=100, max_iter=8000)))                    
  return Pipeline(transformer)

def general_gridsearch():
  pipe = classifier_pipe()
  
  # print(pipe.get_params().keys())
  param_grid = dict(clf__C=[1,100,1000])
  # param_grid = dict(FeatureUnion__content_tfidf__tfidf__ngram_range=[(1,9)])
  # param_grid = dict(clf__solver=['liblinear','sag','newton-cg','lbfgs'])
                      # FeatureUnion__content_tfidf__tfidf__lowercase=[True,False],
  grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=2)
  grid_search.fit(X_train, y_train)
  print(pipe)
  print("score = %3.2f" %(grid_search.score(X_test, y_test)))
  print(grid_search.best_params_)

  y_score = grid_search.decision_function(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, y_score)
  roc_auc = roc_auc_score(y_test, y_score)
  predictions = grid_search.predict(X_test)
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
# print(log_pipe().get_params().keys())

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
