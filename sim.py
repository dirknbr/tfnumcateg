
import numpy as np
from scipy.special import expit

def sim(n=2000, type='regression'):
  # type can be regression or classfication
  num = np.random.normal(0, 1, (n, 3))
  categ1 = np.random.choice(np.arange(1, 11), n)
  categ2 = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f'], n)
  beta = np.random.normal(0, 1, 3)
  xbeta = num.dot(beta) + (categ1 == 1) + (categ2 == 'b') - (categ2 == 'c')
  if type == 'regression':
    return xbeta + np.random.normal(0, 1, n), num, categ1, categ2
  else:
    return np.random.binomial(1, expit(xbeta)), num, categ1, categ2 

if __name__ == '__main__':
  print(sim(type='classif'))