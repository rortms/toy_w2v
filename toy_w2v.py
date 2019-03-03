
# coding: utf-8

# In[1]:


import random
import time
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

## 0.75 power of fraction taken from word 2 vec
words = Counter({ w : int( (count/m)**0.75 )  for w, count in words.most_common() if np.random.uniform() >= P(count/total_words)})


# Word to id mappings
word2int = { tup[0] : i for i, tup in enumerate(words.most_common()) }
int2word = { i : word for word, i in word2int.items() }

print(len(words))


# In[7]:


# Skip-gram model: for each word, sample a surrounding word within a fixed window (skip-window) excluding itself,
#  each word is processed in this way k times (skip number) generating k training targets for it. 
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


# In[14]:


class littleNN(object):

    def __init__(self, word_id_counts, embedding_dim, neg_sample_size, learning_rate = 0.05):

        ## Vocabulary
        self.vocab_size = len(word_id_counts)
        self.unigram_table = list(word_id_counts.elements()) # immitating original w2v
        
        ## Network parameters
        self.embedding_dim = embedding_dim
        self.neg_sample_size = neg_sample_size
        
        # layers
        self.w0 = np.random.normal(0, self.vocab_size**-0.5, (self.vocab_size, embedding_dim ) )
        self.b0 = np.zeros(embedding_dim)
        
        self.w1 = np.random.normal(0, self.vocab_size**-0.5, (embedding_dim, self.vocab_size) )
        self.b1 = np.zeros(self.vocab_size)
        
        # Hyperparams
        self.lr = learning_rate
        
    
    # Negative sampling
    def negSample(self, target_id):

        neg_samples = np.zeros(self.neg_sample_size+1, dtype=np.int32)
        for k in range(self.neg_sample_size):

            # Imitating unigram table idea from original w2v, see;
            # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
            
            # neg_word_id = self.unigram_table[np.random.randint(len(self.unigram_table))]
            # neg_samples[neg_word_id] = 0 # one-hot encoding to 0 
            neg_samples[k] = self.unigram_table[np.random.randint(len(self.unigram_table))]

        neg_samples[-1] = target_id
        return neg_samples

    #
    def crossEntropy(self, softmax_ps, target_id):
        
        return -np.log(softmax_ps[target_id])

    # Softmax
    def Softmax(self, v):

        # Numerical stability, (avoiding large number overflow)
        C = max(v)

        expV = np.exp(v-C)
        return expV / np.sum(expV, axis=1, keepdims=True)
    
    # 
    def forwardPass(self, inBatch):

        # Input to hidden is just a lookup (b/c of one hot word encoding)
        z0 = np.array( [self.w0[word_id] + self.b0 for word_id in inBatch] )
        a0 = np.maximum(0, z0) # ReLu activation

        # Hidden to output
        z1 = np.dot(a0, self.w1) + self.b1
        softmax_ps = self.Softmax(z1)

        return a0, softmax_p

    #
    def backpropUpdate(self, a0, softmax_p, Xbatch, target_batch):

        N = softmax_p.shape[0]
        softmax_ps_d = softmax_p
        softmax_p_d[range(N), target_batch] -= 1
        softmax_p_d /= N # weight updates will be averaged over batch size

        ## w1
        w1_D = np.(a0.T, softmax_p_d)
        b1_d = np.sum(softmax_p_d, axis=0)
        # update
        self.w1 += -self.lr * w1_D
        self.b1 += -self.lr * b1_d

        
        ## w0
        # Treating Xbatch as one-hot implies w1_D consists of copies of
        # a0_d as rows, for every xi = 1, and a row of zero otherwise
        
        a0_d = np.dot(softmax_p_d, self.w1.T)
        b0_d = np.sum(a0_d, axis=0)
        
        # one-hot equivalent of np.dot(Xbatch.T, a0_d) combined with
        # update step

        self.w1[Xbatch] += -self.lr * a0_d
        
    #
    def sampledBackProp(self, inword_id, a0, z1, target_id):

        # Sample
        sampled_indices = self.negSample(target_id)
        sampled_z = z1[sampled_indices]
        # Softmax 
        C = np.max(sampled_z)
        D = np.sum(np.exp(sampled_z -C))
        sampled_softmax = { idx : np.exp(sampled_z[k] - C) for k, idx in enumerate(sampled_indices) }
        
        #
        # Hidden to output calculations
        w0_error = np.zeros(self.embedding_dim)
        delta_weights1= {} # dictionary lookup to update only affected terms
        
        # Deltas for w1
        for kth, row in enumerate(self.w1):

            sm = 0
            for idx in sampled_indices:

                ##########################################
                # 'local' gradient of L is pi-1 when i==target else pi
                
                d = sampled_softmax[idx] if idx != target_id else sampled_softmax[idx] - 1
                delta_weights1[ (kth, idx) ] = d*a0[kth]
                
                ##############

                # 'dot' product on transpose of w1 elementwise and only for sample ids
                sm += row[idx] * d
                
            w0_error[kth] = sm

        ## Input to hidden calculations
        # Note:
        # one-hot encoded input means only kth row of w0
        # will be updated with non-zero deltas, since input xk is coeficient.

        # b/c of one-hot input this happens to be a row vector
        delta_weights0 = w0_error * 1 #* a0 * (1 - a0)

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

        # Update w1
        for key in deltas1:
            i,j = key

            self.w1[i][j] += -self.lr * deltas1[key]

            # update b1
            self.b1[j] += -self.lr *deltas1[key]
            
        # Update w0 
        row_index, deltas0 = deltas0
        self.w0[row_index] += -self.lr * deltas0
        #update b0
        self.b0 += -self.lr * deltas0

    ##
    def word2vec(self, in_word_id):

        return self.w0[in_word_id]
        


# In[15]:


def unit(v):
    return v / np.sqrt(np.dot(v,v))

def cosineDistance(v,w):
    return np.dot( unit(v), unit(w) )


# In[ ]:


def trainNN(word_id_counts):

    # Training Params
    epochs = 30
    batch_size = 1000
    
    # Hyperparams
    embedding_dim = 128
    neg_sample_size = 64
    learning_rate = 0.2
    momentum = 0.5

    # word2vec and validation
    radius = 2      # sample window radius
    repeat_num = 5  # number of times to sample each word
    head_subset_size = 100 # sample only from head of distribution for monitoring progress
    validation_sample_size = 6
    
    # Data iterator
    moreData = iterData(settings["data_path"], batch_size) # data iterator
    
    # Init NN
    net = littleNN(word_id_counts, embedding_dim, neg_sample_size, learning_rate)

    ###############
    # Random sample from dictionary to observe closest cosine distances
    head_of_distribution = [ id_count[0] for id_count in word_id_counts.most_common( head_subset_size )]
    samp_ids = random.sample(head_of_distribution, validation_sample_size)
    pS = pd.Series([w_id for w_id in word_id_counts.keys() if w_id not in samp_ids])
        
    ##############
    #
    initw = net.w0
    initb = net.b0
    for e in range(epochs):
        
        # for v in [net.word2vec(i) for i in samp_ids]:
        #     print(v)
        # print("-------------------\n")
        
        
        ## View sampled words' neighbors every epoch
        for samp_id in samp_ids:
            neighbor_ids = pS.apply( lambda w_id: cosineDistance(net.word2vec(samp_id), net.word2vec(w_id)) ).nlargest(6).index.values
            print("Words close to {}: {}".format(int2word[samp_id], [int2word[i] for i in neighbor_ids] ) )
        print("------------\n")
        

        ## Train ##
        
        batch = next(moreData)
        words, targets = inputsTargets(batch, radius=radius, repeat_num=repeat_num ) 
        average_cost = 0
        
        for nth, word in enumerate(words):
            
            word_id, target_id  = word2int.get(word,None), word2int.get(targets[nth], None)

            if not word_id or not target_id:
                continue
            
            a0, softmax_probs = net.forwardPass(word_id)
            # print (np.mean(softmax_probs))
            d0, d1 = net.sampledBackProp(word_id, a0, softmax_probs, target_id)
            net.updateWeights(d0,d1)

            # 
            ce = net.crossEntropy(softmax_probs, target_id)
            if ce:
                average_cost += ce

        #
        
        # print( sum ( sum( abs(np.abs(net.w0) - initw) ) ) ) 
        
        # print(sum(abs(abs(net.b0) - initb)))
        
        print(average_cost / len(words))

            
## Run training
word_id_counts = Counter( { word2int[w] : words[w] for w in words } )

trainNN(word_id_counts)


# In[ ]:


pS = pd.Series(np.array([1,2,3,4,5,74,7,78]))
pS.apply(lambda x: 1/x).nsmallest(3).index.values


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

