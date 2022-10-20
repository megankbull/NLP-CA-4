import sys
import numpy as np

RESULTS_FILE = 'percepoutput.txt'
ANS_KEY = 'perceptron-training-data/dev-key.txt'

def read_file(file_name):
  try:
    with open(file_name, 'r') as file:
      lines = file.readlines()
      results = np.zeros((len(lines), 2))
      for i, line in enumerate(lines):
        uid, tf, pn = line.strip().split(' ')
        results[i, 0] = -1 if tf == 'Fake' else 1
        results[i, 1] = -1 if pn == 'Neg' else 1
      return results
  except FileNotFoundError:
    print('Error: file', file_name, 'not found')

def get_acc(preds, trues):
  total = len(preds)
  tf_corr = 0.
  pn_corr = 0.
  for pred, true in zip(preds, trues):
    if pred[0] == true[0]:
      tf_corr += 1
    if pred[1] == true[1]:
      pn_corr += 1
  
  return tf_corr/total, pn_corr/total

def main():

  preds = read_file(RESULTS_FILE)
  trues = read_file(ANS_KEY)

  tf_acc, pn_acc = get_acc(preds, trues)
  print('\tTrue/Fake Accuracy:', tf_acc)
  print('\tPos/Neg Accuracy:', pn_acc)

if __name__ == '__main__':
  main()