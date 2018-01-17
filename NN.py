import sys
import pandas as pd
import numpy as np
import tensorflow as tf


data = pd.read_csv("/home/rik/train.csv")

data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'][data['Embarked']=='C']=0.0
data['Embarked'][data['Embarked']=='Q']=0.5
data['Embarked'][data['Embarked']=='S']=1.0
data['Sex'][data['Sex']=='male']=0.0
data['Sex'][data['Sex']=='female']=1.0
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].median())

'''data['Age'] = (data['Age'] - data['Age'].min())/(data['Age'].max() -data['Age'].min())
data['Fare'] = (data['Fare'] - data['Fare'].min())/(data['Fare'].max() -data['Fare'].min())
data['Parch'] = (data['Parch'] - data['Parch'].min())/(data['Parch'].max() -data['Parch'].min())
data['SibSp'] = (data['SibSp'] - data['SibSp'].min())/(data['SibSp'].max() -data['SibSp'].min())
data['Pclass'] = (data['Pclass'] - data['Pclass'].min())/(data['Pclass'].max() -data['Pclass'].min())
'''
features = data[['Age', 'Fare', 'Parch', 'SibSp', 'Embarked', 'Pclass', 'Sex']]
lbls = data[['Survived']]

features = features
lbls = lbls

mu = np.mean(features, axis=0)
sigma = (np.std(features, axis=0))
features = (features - mu) / sigma

n_examples = len(lbls)

# Model

# Hyper parameters

epochs = 100
learning_rate = 0.01
batch_size = 5

input_data = tf.placeholder('float', [None, 7])
labels = tf.placeholder('float', [None, 1])

weights = {
      'hl1': tf.Variable(tf.random_normal([7, 10])),
      'hl2': tf.Variable(tf.random_normal([10, 10])),
      'hl3': tf.Variable(tf.random_normal([10, 7])),
      'ol': tf.Variable(tf.random_normal([7, 1]))
      }

biases = {
      'hl1': tf.Variable(tf.zeros([10])),
      'hl2': tf.Variable(tf.zeros([10])),
      'hl3': tf.Variable(tf.zeros([7])),
      'ol': tf.Variable(tf.zeros([1]))
      }



hl1 = tf.nn.relu(tf.add(tf.matmul(input_data, weights['hl1']), biases['hl1']))
hl2 = tf.nn.relu(tf.add(tf.matmul(hl1, weights['hl2']), biases['hl2']))
hl3 = tf.nn.relu(tf.add(tf.matmul(hl2, weights['hl3']), biases['hl3']))
ol = tf.nn.sigmoid(tf.add(tf.matmul(input_data, weights['ol']), biases['ol']))

loss = tf.reduce_mean((labels - ol)**2)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iterations = int(n_examples/batch_size)


def training_accuracy():
  foo,  = sess.run([ol], feed_dict={input_data: features, labels: lbls})
  return (float(np.count_nonzero(np.equal(np.round(foo), lbls))) / float(lbls.shape[0]))


print("Initial training accuracy %f" % training_accuracy())


for epoch_no in range(epochs):
  ptr = 0
  for iteration_no in range(iterations):
    epoch_input = features[ptr:ptr+batch_size]
    epoch_label = lbls[ptr: ptr+batch_size]
    ptr = (ptr + batch_size)%len(features)
    _, err = sess.run([train, loss], feed_dict={input_data: epoch_input, labels: epoch_label})
  print("Error at epoch ", epoch_no, ": ", err)
  print("  Training accuracy %f" % training_accuracy())


test = pd.read_csv("/home/rik/test.csv")
datat = test[['Age', 'Fare', 'Parch', 'SibSp', 'Embarked', 'Pclass', 'Sex']]
datat['Age'] = datat['Age'].fillna(datat['Age'].median())
datat['Embarked'][datat['Embarked']=='C']=0.0
datat['Embarked'][datat['Embarked']=='Q']=0.5
datat['Embarked'][datat['Embarked']=='S']=1.0
datat['Sex'][datat['Sex']=='male']=0.0
datat['Sex'][datat['Sex']=='female']=1.0
datat['Fare'] = datat['Fare'].fillna(datat['Fare'].median())

mu = np.mean(datat, axis=0)
sigma = (np.std(datat, axis=0))
features = (datat - mu) / sigma


res,  = sess.run([ol], feed_dict={input_data: datat})
out = pd.read_csv('/home/rik/test.csv', sep = ',', usecols = ['PassengerId'])
out['Survived'] = res
out.to_csv('newsub.csv', index = False)
