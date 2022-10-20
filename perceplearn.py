# python perceplearn.py /path/to/input

import sys 
import json 
import string
import numpy as np 

REVIEWS = []
INPUTS = []
AVG_EPOCHS = 12
VAN_EPOCHS = 21
WORD_SET = set()

def read_file(f_path=sys.argv[1]):
   global REVIEWS, WORD_SET, REAL_DS, SENTIM_DS

   with open(f_path, 'r') as f: 
      for l in f:
         spl_l = l.split(' ')
         REVIEWS.append((spl_l[0], 
                        label_real(spl_l[1]), 
                        label_sentim(spl_l[2]), 
                        clean_text(spl_l[3:])))

   WORD_SET = list(WORD_SET)

def label_real(text): 
    
   return 1 if text=="True" else -1 

def label_sentim(text): 
    
   return 1 if text=="Pos" else -1 

def clean_text(text_list): 
   global WORD_SET
   text = " ".join(text_list).strip()
   
   new_text = text.translate(str.maketrans(string.punctuation,  ' '*len(string.punctuation))).lower()
   new_text = new_text.translate(str.maketrans(string.digits, ' '*len(string.digits)))

   toks = new_text.split()
   WORD_SET.update(toks) 
   return toks

def featurize_text():
   global REVIEWS, WORD_SET, INPUTS

   for sample in REVIEWS: 
      vec = np.zeros(len(WORD_SET), dtype=np.float64)
      toks = sample[3]
      for i, w in enumerate(WORD_SET): 
         vec[i] = toks.count(w)
      INPUTS.append((vec, sample[1], sample[2]))

def train_van(): 
   global VAN_EPOCHS, INPUTS

   s_w = np.zeros(INPUTS[0][0].size, dtype=np.float64)
   s_b = 0 # bias 
   
   r_w = np.zeros(INPUTS[0][0].size, dtype=np.float64)
   r_b = 0 # bias  
   
   for _ in range(VAN_EPOCHS):
      for x,y,z in INPUTS: 
         a_s = np.sum(s_w * x) + s_b
         a_r = np.sum(r_w * x) + r_b

         if z*a_s <= 0:
            s_w = np.add(s_w, z*x)
            s_b += z

         if y*a_r <= 0:
            r_w = np.add(r_w, y*x)
            r_b += y

   return (s_w, s_b), (r_w, r_b)

def train_avg(): 
   global AVG_EPOCHS, INPUTS

   s_w = np.zeros(INPUTS[0][0].size, dtype=np.float64)
   s_b = 0 # bias 
   s_cache_w = np.zeros(INPUTS[0][0].size, dtype=np.float64)
   s_cache_b = 0
   
   r_w = np.zeros(INPUTS[0][0].size, dtype=np.float64)
   r_b = 0 # bias
   r_cache_w = np.zeros(INPUTS[0][0].size, dtype=np.float64)
   r_cache_b = 0 # bias  

   c = 1
   for _ in range(AVG_EPOCHS):
      for x,y,z in INPUTS: 

         a_s = np.sum(s_w * x) + s_b
         a_r = np.sum(r_w * x) + r_b

         if z*a_s <= 0:
            s_w = np.add(s_w, z*x)
            s_b += z
            s_cache_w += z*c*x
            s_cache_b += z*c

         if y*a_r <= 0:
            r_w = np.add(r_w, y*x)
            r_b += y
            r_cache_w += y*c*x
            r_cache_b += y*c

         c += 1

   s_avg_w = s_w - (1/c * s_cache_w)
   s_avg_b = s_b - (1/c * s_cache_b)

   r_avg_w = r_w - (1/c * r_cache_w)
   r_avg_b = r_b - (1/c * r_cache_b)

   return (s_avg_w, s_avg_b), (r_avg_w, r_avg_b)

def write_model(s_model, r_model, output): 
   global WORD_SET

   np.set_printoptions(threshold=sys.maxsize)
   s_w_str = np.array2string(s_model[0])
   r_w_str = np.array2string(r_model[0])

   with open(output, 'w+') as f: 
      f.write(f"Sentiment Classifier\n\nbias: {s_model[1]}\n\nWeights:\n\n{s_w_str}\n\n")
      f.write(f"Real Classifier\n\nbias: {r_model[1]}\n\nWeights:\n\n{r_w_str}\n\n")

      f.write(f"unique words:\n\n{json.dumps(WORD_SET)}")

def train_models(): 
   read_file()
   featurize_text()
      
   s_van_model, r_van_model = train_van()
   write_model(s_van_model, r_van_model, 'vanillamodel.txt')

   s_avg_model, r_avg_model = train_avg()
   write_model(s_avg_model, r_avg_model, 'averagedmodel.txt')

def main(): 
   train_models()

if __name__ == '__main__': 
   main()
