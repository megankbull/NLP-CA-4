# python percepclassify.py /path/to/model /path/to/input
import numpy as np 
import sys 
import string
import json 

SAMPLES = []
WORD_SET = None

S_WEIGHTS = None 
S_BIAS = None 

R_WEIGHTS = None
R_BIAS = None 

def read_data(f_path=sys.argv[2]):
   global SAMPLES

   with open(f_path, 'r') as f: 
      for l in f:
         spl_l = l.split(' ')
         SAMPLES.append((spl_l[0], featurize_text(spl_l[1:])))

def clean_text(text_list): 
   text = " ".join(text_list).strip()
   
   new_text = text.translate(str.maketrans(string.punctuation,  ' '*len(string.punctuation))).lower()
   new_text = new_text.translate(str.maketrans(string.digits, ' '*len(string.digits)))

   return new_text.split()

def featurize_text(text_list):
   global WORD_SET
   toks = clean_text(text_list)

   vec = np.zeros(len(WORD_SET), dtype=np.float64)
   
   for i, w in enumerate(WORD_SET): 
      vec[i] = toks.count(w)
   
   return vec
   
def get_label(classif, num): 
   if classif=='s': return 'Pos' if num > 0 else 'Neg'
   else: return 'True' if num > 0 else 'Fake'

def classify_input(): 
   global SAMPLES, S_WEIGHTS, S_BIAS, R_WEIGHTS, R_BIAS

   tagged = []
   for id, feat in SAMPLES: 
      s = np.sum(S_WEIGHTS * feat) + S_BIAS
      r = np.sum(R_WEIGHTS * feat) + R_BIAS
      tagged.append((id, get_label('r', r), get_label('s', s)))
   return tagged

def write_output(f_path='percepoutput.txt'): 
   tagged = classify_input()

   with open(f_path, 'w+') as f: 
      for s in tagged: 
         f.write(f"{s[0]} {s[1]} {s[2]}\n")
   
def load_model(f_path=sys.argv[1]): 
   global WORD_SET, S_WEIGHTS, S_BIAS, R_WEIGHTS, R_BIAS

   with open(f_path, "r") as f: 
      raw_params = f.read().split("\n\n")

      S_BIAS = float(raw_params[1].partition(" ")[-1])
      S_WEIGHTS = np.fromstring(raw_params[3][1:-1], sep=' ')
      
      R_BIAS = float(raw_params[5].partition(" ")[-1])
      R_WEIGHTS = np.fromstring(raw_params[7][1:-1], sep=' ')
      WORD_SET = json.loads(raw_params[9])

def main(): 
   load_model()
   read_data()
   write_output()

if __name__ == '__main__': 
   main()