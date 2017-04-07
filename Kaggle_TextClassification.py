import sys
import numpy
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def preprocessing(path, filename, test_flag=False):
    df1=pd.read_csv(path, sep='\t')
    if test_flag:
        file = open(filename, 'w')
        for i, r in df1.iterrows():
            r[0]=r[0].replace(',',' ')
            file.write(''.join(r[0].split('0', 1)))
            file.write('\n')
        file.close()
    else:
        df1.columns = ['a']
        df1['a'] = df1['a'].map(lambda x: x.rstrip(','))
        df1['a'] = df1['a'].str.replace(r', ',',')
        file = open(filename, 'w')
        for i, r in df1.iterrows():
            file.write(','.join(r[0].split(',')))
            file.write('\n')
        file.close()

def train(train_path, filename):
    preprocessing(train_path, filename)
    X_train, Y_train = load_svmlight_file(filename, multilabel=True)
    Y = MultiLabelBinarizer().fit_transform(Y_train)
    m = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y)
    return [m,X_train]

def test(test_path, model, x_train):
    data = []
    preprocessing(test_path, 'test2.txt',test_flag=True)
    X_test, Y_test = load_svmlight_file('test2.txt',n_features=x_train.shape[1],multilabel=False)
    result = model.predict(X_test)
    for i in result:
        temp=[]
        for j in i.nonzero():
            for c in j:
                temp.append(c)
        data.append(", ".join(str(x) for x in temp))
    df = pd.DataFrame(data)
    df.reset_index(level=0, inplace=True)
    df.to_csv('prediction.csv', index=False)
    #print numpy.nonzero(m[0])
    # indices = zip(*m.nonzero())
    # print indices
def main():
    model,train_X = train('C:\Kaggale\\train-remapped-000.000.000.000.csv', 'train.txt')
    test('C:\Kaggale\\test-remapped\\samples.csv', model, train_X)


main()


#indices = zip(*m.nonzero())