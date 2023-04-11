"""
TF model with embedding for categorical variables
regression or classification

https://github.com/google/lifetime_value/blob/dd418967b3ca456b375b0ae8adac38732f7831db/notebooks/kaggle_acquire_valued_shoppers_challenge/classification.ipynb

https://mmuratarat.github.io/2019-06-12/embeddings-with-numeric-variables-Keras


"""
import tensorflow as tf
import numpy as np
from sklearn import preprocessing, metrics
from sim import *

class TfNumCateg:
  def __init__(self, categ_inputs, categ_outputs, num_inputs, type='regression', 
               lr=.001):
    """TF model with numerical and categorical variables.
    Args:
      categ_inputs: list of number of input categories
      categ_outputs: list of number of embedding dimensions per variable
      num_inputs: number of numeric inputs
      type: regression or classification.
    """
    assert len(categ_inputs) == len(categ_outputs)
    self.categ_inputs = categ_inputs
    self.categ_outputs = categ_outputs
    if type == 'classification':
      actfunc = 'sigmoid'
      loss = 'bce'
    else:
      actfunc = 'linear'
      loss = 'mse'
    embeds = []
    for cid in range(len(categ_inputs)):
      embeds.append(tf.keras.Sequential([
        tf.keras.layers.Embedding(categ_inputs[cid], categ_outputs[cid]),
        tf.keras.layers.Flatten()]))
    num_input = tf.keras.layers.Input(shape=(num_inputs,))
    embed_input = [tf.keras.layers.Input(shape=(1,), dtype=np.int64) for _ in range(len(embeds))]
    embed_output = [embeds[cid](embed_input[cid]) for cid in range(len(embeds))]
    deep_input = tf.keras.layers.concatenate([num_input] + embed_output)
    # TODO: expose network parameters
    deep_model = tf.keras.Sequential([
      tf.keras.layers.Dense(12),
      tf.keras.layers.Dense(8),
      tf.keras.layers.Dense(1, activation=actfunc)])
    self.model = tf.keras.Model(inputs=[num_input] + embed_input, outputs=deep_model(deep_input))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    self.model.compile(loss=loss, optimizer=opt)

  def fit(self, num, categ, y, epochs=30):
    self.model.fit([num] + categ, y, epochs=epochs)

  def predict(self, num, categ):
    return self.model.predict([num] + categ)

def categ_to_int(x):
  # return an index variable
  le = preprocessing.LabelEncoder()
  return le.fit_transform(x)

if __name__ == '__main__':
  type = 'classification'
  y, num, categ1, categ2 = sim(type=type)
  categ1_int = categ_to_int(categ1)
  categ2_int = categ_to_int(categ2)
  categ1_len = len(set(categ1_int))
  categ2_len = len(set(categ2_int))
  model = TfNumCateg(categ_inputs=[categ1_len, categ2_len], 
                     categ_outputs=[3, 2], num_inputs=3, type=type)
  model.fit(num, [categ1_int, categ2_int], y)
  # print(model.model.summary())
  pred = model.predict(num, [categ1_int, categ2_int])
  if type == 'classification':
    print(metrics.roc_auc_score(y, pred))
  else:
    print(metrics.mean_absolute_error(y, pred))

