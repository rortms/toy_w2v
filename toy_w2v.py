
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


# Word to id mappings
word2int = { tup[0] : i for i, tup in enumerate(words.most_common()) }
int2word = { i : word for word, i in word2int.items() }

print(len(words))


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


# In[17]:


class littleNN(object):

    def __init__(self, word_id_counts, embedding_dim, neg_sample_size, learning_rate = 0.05):

        ## Vocabulary
        self.vocab_size = len(word_id_counts)
        self.unigram_table = list(word_id_counts.elements()) # immitating original w2v
        
        ## Network parameters
        self.embedding_dim = embedding_dim
        self.neg_sample_size = neg_sample_size        
        
        # layers
        self.w0 = np.random.uniform( size=(self.vocab_size, embedding_dim ) )
        self.w1 = np.random.uniform( size=(embedding_dim, self.vocab_size) )
        
        # sigmoid activation
        self.sgmd = lambda x: 1 / ( 1 + np.exp(-x))
        self.lr = learning_rate
        
    # Softmax
    def Softmax(self, v):

        # Numerical stability, (avoiding large number overflow)
        C = max(v)

        expV = np.exp(v-C)
        return expV / sum(expV)
    
    # Negative sampling
    def negSample(self):

        neg_samples = {}
        for k in range(self.neg_sample_size):

            # Imitating unigram table idea from original w2v, see;
            # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
            neg_word_id = self.unigram_table[np.random.randint(len(self.unigram_table))]
            
            neg_samples[neg_word_id] = 0 # one-hot encoding to 0 

        return neg_samples

    # Forward pass
    def forwardPass(self, inword_id):

        # Input to hidden is just a lookup (b/c of one hot word encoding)
        z0 = self.w0[inword_id]
        a0 = self.sgmd(z0)

        # Hidden to output
        z1 = np.dot(a0, self.w1)
        softmax_ps = self.Softmax(z1)

        return a0, softmax_ps

    # Sampled back-prop
    def sampledBackProp(self, inword_id, a0, softmax_out, target_id):

        sampled_targets = self.negSample()
        sampled_targets[target_id] = 1 # one-hot for target is 1


        ## Sampled backprop ##

        # Hidden to output calculations
        w0_error = np.zeros(self.embedding_dim)
        delta_weights1= {} # dictionary lookup to update only affected terms
        
        for kth, row in enumerate(self.w1):

            sm = 0
            for idx in sampled_targets:
                
                out_err_at_idx = softmax_out[idx] - sampled_targets[idx]
                
                ###############################################
                # 'Local' gradient for softmax node
                if kth == idx:
                    softmax_derivative = softmax_out[idx] * (1 - softmax_out[idx] )
                else:
                    softmax_derivative = - softmax_out[idx] * softmax_out[kth]
                ###############################################
                
                delta_weights1[ (kth,idx) ] = a0[kth] * softmax_derivative * out_err_at_idx

                # Dot transpose of w1
                sm += row[idx] * softmax_derivative * out_err_at_idx

            w0_error[kth] = sm

        ## Input to hidden calculations
        # Note:
        # one-hot encoded input means only kth row of w0
        # will be updated with non-zero deltas, since input xk is coeficient.

        # b/c of one-hot input this happens to be a row vector
        delta_weights0 = a0 * (1 - a0) *  w0_error * 1

        return (inword_id, delta_weights0), delta_weights1
    ##
    def updateWeights(self, deltas0, deltas1):
        '''
        Inputs
        ------
        deltas0: 
        A tuple, (index, delta_vector) index of w0 row to be updated and deltas
        
        deltas1:
        Dictionary, keys are tuples corresponding to entry i,j of w1 to be updated.
        Values are the deltas at index i,j
        
        '''
        # Update w0 
        row_index, deltas0 = deltas0
        self.w0[row_index] -= self.lr * deltas0

        # Update w1
        for key in deltas1:
            i,j = key

            self.w1[i][j] -= self.lr * deltas1[key]

    ##
    def word2vec(self, in_word_id):

        return self.w0[in_word_id]
        


# In[ ]:


def trainNN(word_id_counts, epochs, batch_size):

    # Hyperparams
    embedding_dim = 400
    neg_sample_size = 15
    learning_rate = 0.5
    moreData = iterData(settings["data_path"], batch_size)
    # word2vec specific
    radius = 4      # sample window radius
    repeat_num = 3  # number of times to sample each word
    
    #
    net = littleNN(word_id_counts, embedding_dim, neg_sample_size, learning_rate)
    
    #
    for e in range(epochs):
        
        batch = next(moreData)
        words, targets = inputsTargets(batch)

        for nth, word in enumerate(words):
            
            word_id, target_id  = word2int.get(word,None), word2int.get(targets[nth], None)

            if not word_id or not target_id:
                continue
            
            a0, softmax_probs = net.forwardPass(word_id)
            d0, d1 = net.sampledBackProp(word_id, a0, softmax_probs, target_id)
            net.updateWeights(d0,d1)


## Run training             
word_id_counts = Counter( { word2int[w] : words[w] for w in words } )

trainNN(word_id_counts, 5, 10000)


# In[18]:




testN = littleNN(word_id_counts, 300, 5, )

word_id = word2int['weather']
targ_id = word2int['wind']

a0, sfm = testN.forwardPass(word_id)
d0,d1 = testN.sampledBackProp(word_id, a0, sfm, targ_id)
testN.updateWeights(d0,d1)


# In[5]:


t = np.array([1,2,3,4,5,74,7,78])
t * (1 - t)

