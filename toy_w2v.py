
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
import itertools as itt


# In[2]:


# Infinite data iterator
def iterData(filename, batch_size, jump_around=False):

    # Iterator reading file in batches
    def get_batch():
    
        with open(filename, "r") as datfile:

            position, offset = 0, batch_size

            while True:
                
                if jump_around:
                    offset = np.random.randint(1,4) * batch_size
                    
                yield datfile.read(batch_size)
                position = datfile.seek(position + offset)

    batch = get_batch()

    # When file is exhausted, start over with new get_batch iterator
    while True:
        b = next(batch)
        if b:
            yield  b.split()
        else:
            print("EOF, starting over")
            batch = get_batch()
        


# In[3]:


# Test infinite iterator
fn = "./testf"

with open(fn, "r") as tf:
    lines = tf.read()
print(len(lines))

# Only 54 charactors reused indefinitely
testIter = iterData(fn, 5)
print(list( (" ".join(next(testIter)) for k in range(100)) ) )


# In[4]:


## Initialize file iterator
from config import settings

moreData = iterData(settings["data_path"], 1000)


# In[5]:


# Construct vocabulary set
words = Counter(next(moreData))
print(len(words))
for k in range(70000):
    words.update(next(moreData))
    if k % 3000 == 0:
        print(len(words))
        


# In[6]:


# Subsampling with word2vec's subsampling function
# following: http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

# Increase the chance of a less common word being kept, and reduce that of very common words
def P(word_fraction):

    return np.sqrt(1e-7/word_fraction) + 1e-7/word_fraction

print(len(words))

m = min(words.values())

total_words = sum(words.values())

words = Counter({ w : int( (count/m)**0.75 )  for w, count in words.most_common() if np.random.uniform() >= P(count/total_words)})

print(len(words))


# In[7]:


# Word to id mappings
word2int = { tup[0] : i for i, tup in enumerate(words.most_common()) }
int2word = { i : word for word, i in word2int.items() }


# In[8]:


# Skip-gram model: for each word, sample a surrounding word within a fixed window (excluding itself),
# (Forgot actual terminology) each word is processed in this way k times generating k training targets for it.
#

def inputsTargets(word_sequence, radius = 4, repeat_num = 2):

    words, targets = [None]*len(word_sequence)*repeat_num, [None]*(len)(word_sequence)*repeat_num
    batch_size = len(words) # or targets
    
    for i in range(0,batch_size, repeat_num):
        
        # Index in word_sequence array ( len(word_sequence) < batch_size )
        index = i // repeat_num
        word = word_sequence[index]
        
        lower = 0 if index < radius else index - radius
        upper = index + radius
        words_in_window = word_sequence[lower:index] + word_sequence[index+1:upper]
        
        for k in range(repeat_num):
            words[i+k] = word
            targets[i+k] = words_in_window[np.random.randint(0,len(words_in_window))]

    return words, targets


# In[ ]:


class littleNN(object):

    def __init__(self, word_counts, embedding_dim, neg_sample_size):

        #Network parameters
        self.input2hidden = np.random.uniform(

    


# In[30]:


testN = littleNN(words, 300, 5)
print(testN.forward_pass(word2int['weather']))
testN.negSample()


# In[9]:


t = np.array([1,2,3,4,5,74,7,78])
t[[0,5,3]]

