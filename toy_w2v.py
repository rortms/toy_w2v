
# coding: utf-8

# In[1]:


import random
import time

import numpy as np
from scipy.sparse import csr_matrix
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


# # Vocab
# with open(settings["word_filter"], 'r') as fw:
#     filter_words = fw.read()
    
# with open(settings["data_path"], 'r') as wordfile:
#     words = wordfile.read()

# filter_words = Counter(filter_words.split())
# words = Counter( [ w for w in words.split() if filter_words.get(w, None) is None] )


# In[7]:


print(len(words))


# In[8]:


# Subsampling with word2vec's subsampling function
# following: http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

# Increase the chance of a less common word being kept, and reduce that of very common words
def P(word_fraction):

    return np.sqrt(1e-7/word_fraction) + 1e-7/word_fraction

print(len(words))

m = min(words.values())

total_words = sum(words.values())

## 0.75 power of fraction taken from word 2 vec
words = Counter({ w : int( (100*count)**0.75 )  for w, count in words.most_common() if np.random.uniform() >= P(count/total_words)})


# Word to id mappings
word2int = { tup[0] : i for i, tup in enumerate(words.most_common()) }
int2word = { i : word for word, i in word2int.items() }

print(len(words))


# In[10]:


###################
# Skip-gram model: for each word, sample a surrounding word within a fixed window (skip-window) excluding itself,
# each word is processed in this way k times (skip number) generating k training targets for it. 
#

def makeInputsTargets(word_sequence, radius = 4, repeat_num = 2):

    words = np.zeros(len(word_sequence)*repeat_num, dtype=np.int32)
    targets =  np.zeros((len)(word_sequence)*repeat_num, dtype=np.int32 )
    batch_size = len(words) # or targets
    
    for i in range(0, batch_size, repeat_num):
        
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


# In[23]:


class littleNN(object):

    def __init__(self, word_id_counts, embedding_dim, neg_sample_size, learning_rate = 0.5):

        ## Vocabulary
        self.vocab_size = len(word_id_counts)
        self.unigram_table = list(word_id_counts.elements()) # immitating original w2v
        
        ## Network parameters
        self.embedding_dim = embedding_dim
        self.neg_sample_size = neg_sample_size
        
        # layers
        self.w0 = np.random.normal(0, self.vocab_size**-0.5, (self.vocab_size, embedding_dim ) )
        self.b0 = np.zeros(embedding_dim)
        # For softmax implementation
        self.w1 = np.random.normal(0, self.vocab_size**-0.5, (embedding_dim, self.vocab_size) )
        self.b1 = np.zeros(self.vocab_size)
        # For negative sampling implementation
        self.u = np.random.normal(0, self.vocab_size**-0.5, (embedding_dim, 1) )
        self.b = np.random.normal(0, self.vocab_size**-0.5)
        
        # Hyperparams
        self.lr = learning_rate
        self.reg = 1e-3

        # Activations
        self.sgmd = lambda x: 1 / (1 + np.exp(-x)) # Sigmoid

        
    # Softmax
    def Softmax(self, v):

        # Numerical stability, (avoiding large number overflow)
        C = np.max(v) / 4

        expV = np.exp(v-C)
        return expV / np.sum(expV, axis=1, keepdims=True)
    
    # Negative sampling
    def negSample(self, target_id):

        target_id = target_id[0]
        neg_samples = np.zeros(self.neg_sample_size+1, dtype=np.int32)
        for k in range(self.neg_sample_size):

            # Imitating unigram table idea from original w2v, see;
            # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
            
            # neg_word_id = self.unigram_table[np.random.randint(len(self.unigram_table))]
            # neg_samples[neg_word_id] = 0 # one-hot encoding to 0
            samp = self.unigram_table[np.random.randint(len(self.unigram_table))]
            if samp != target_id:
                neg_samples[k] = samp
            else:
                neg_samples[k] = self.unigram_table[np.random.randint(len(self.unigram_table))]

        neg_samples[-1] = target_id
        return neg_samples
    
    #
    def forwardPass(self, Xbatch):

        # Input to hidden is just a lookup (b/c of one hot word encoding)
        # z0 = np.array( [self.w0[word_id] + self.b0 for word_id in inBatch] )
        z0 = self.w0[Xbatch] + self.b0
        a0 = np.maximum(0, z0) # ReLu activation

        # Hidden to output
        z1 = a0 @ self.w1 + self.b1
        return a0, z1

    # def forwardPass(self, Xbatch):
        
    #     # Input to hidden is just a lookup (b/c of one hot word encoding)
    #     # z0 = np.array( [self.w0[word_id] + self.b0 for word_id in inBatch] )
    #     z0 = self.w0[Xbatch] + self.b0
    #     a0 = np.maximum(0, z0) # ReLu activation

    #     # Hidden to output
    #     z1 = np.dot(a0, self.u) + self.b

    #     return a0, z1 
    
    #
    def sampledBackProp(self, a0, z1, Xbatch, target_batch):

        N = len(Xbatch)
        
        # Last element contains target/positive index (see negSample above). All others considered negative
        sample_indices = np.apply_along_axis(self.negSample, 1, target_batch[:,None]) # See explanatory ex1 below

        z1 = z1[np.arange(N)[:,None], sample_indices] # See explanatory ex2 below
        out = self.sgmd(z1)

        n = z1.shape[1] # "binary" classification occurs along sample size, not batch size
        # Sigmoid error
        out_d = -out / n
        out_d[:, -1] += 1 / n  # i.e err = (y - sgmd(z1)) where y = 0,1
        
        # Create sparse csr matrix
        rows = np.array([ [ k for _ in range(len(sample_indices[k])) ] for k in range(N)]).flatten()
        columns = sample_indices.flatten()
        data = out_d.flatten()
        sparse_out_d = csr_matrix( (data, (rows, columns)), shape=(N, self.vocab_size))
                
        ## w1 ##
        
        w1_D = a0.T @ sparse_out_d
        b1_d = np.sum(out_d, axis=0)
        
        w1_D += self.reg * self.w1 # regularization gradient
        
        # update
        self.w1 += -self.lr * w1_D
        self.b1[sample_indices] += -self.lr * b1_d

        
        ## w0 ##

        # Local gradient of ReLu a0 is 1 if z0>0 else 0
        a0_d = sparse_out_d @ self.w1.T
        a0_d[a0 <= 0] = 0
        
        b0_d = np.sum(a0_d, axis=0)

        # Xbatch as sparse matrix
        rows = range(N)
        columns = Xbatch
        data = np.ones(N)
        
        sparse_Xbatch = csr_matrix( (data, (rows, columns)), shape=(N, self.vocab_size))
        w0_D = sparse_Xbatch.T @ a0_d
        
        w0_D += self.reg * self.w0 # regularization gradient
        
        # update
        self.w0 += -self.lr * w0_D
        self.b0 += -self.lr * b0_d
    
    #
    def backpropUpdate(self, a0, z1, Xbatch, target_batch):
        
        softmax_p = self.Softmax(z1)
        N = softmax_p.shape[0]
        softmax_p_d = softmax_p
        softmax_p_d[range(N), target_batch] -= 1
        softmax_p_d /= N # weight updates will be averaged over batch size

        ## w1
        w1_D = np.dot(a0.T, softmax_p_d)
        b1_d = np.sum(softmax_p_d, axis=0)

        w1_D += self.reg * self.w1 # regularization gradient
        
        # update
        self.w1 += -self.lr * w1_D
        self.b1 += -self.lr * b1_d

        ## w0

        # Local gradient of ReLu a0 is 1 if z0>0 else 0
        a0_d = np.dot(softmax_p_d, self.w1.T)
        a0_d[ a0_d <= 0 ] = 0 
        b0_d = np.sum(a0_d, axis=0)
        
        # Treating Xbatch as one-hot implies w1_D consists of copies of
        # a0_d as rows, for every xi = 1, and a row of zero otherwise
        
        # one-hot equivalent of np.dot(Xbatch.T, a0_d) combined with
        # update step.

        for k, index in enumerate(Xbatch):
            self.w0[index] += -self.lr * a0_d[k]
            
        self.b0 += -self.lr * b0_d
    

    #
    def crossEntropy(self, out, target_ids):

        N = out.shape[0]
        return -np.log(out[range(N), target_ids]).sum() / N
    
    #
    def word2vec(self, in_word_id):
        return self.w0[in_word_id]

    # #
    # def getNegSample(self, target):

    #     target = target[0]
    #     result = np.zeros(self.neg_sample_size+1, dtype=np.int32)

    #     # Target splits vocab range in two, pick sample from larger
    #     # excluding the target
    #     if target < (self.vocab_size-1) / 2:

    #         samp = random.sample(range(target+1, self.vocab_size), self.neg_sample_size)
    #     else:
    #         samp = random.sample(range(0, target-1), self.neg_sample_size)

    #     result[:len(samp)] += samp # fill result array
    #     result[-1] = target # add target (assures target exists exactly once)

    #     return result    


# In[24]:


np.array([[1,2,4],[1,1,1]]) *  np.array([[1,0,1],[0,0,2]])


# In[25]:


csr_matrix(2*np.random.randint(0,2, (6,4)).T) @ (4*np.eye(6))


# In[26]:


##################
# Explanatory Ex 1
tars = np.random.randint(0,20, 10)
                
print("targets", tars)
                
def getNegSample(target, N=20):

    target = target[0]
    
    if target < (N-1) / 2:

        return random.sample(range(target+1, N), 4)

    return random.sample(range(0, target-1), 4)

# for each target apply getNegSample function
np.apply_along_axis(getNegSample, 1, tars[:,None])               


# In[27]:


######################
# Explanatory Ex 2

l = 7
ints = np.array([range(11, 3, -1) for k in range(l)])
indices = np.array([[5,0] for k in range(l)])
print(ints)
print(indices)
select_ints = ints[np.arange(l)[:,None],indices]

print(select_ints)


# In[28]:


###################
# Explanatory Ex 3

rows = np.array([ [k for l in range(len(indices[k]))] for k in range(l) ]).flatten()
print("rows", rows)
columns = indices.flatten()
print("columns", columns)
data = select_ints.flatten()

print("data", data)

csr = csr_matrix( (data, (rows, columns)), shape=(7,8))
print(csr.toarray())


# In[29]:


help(csr_matrix.dot)


# In[30]:


def cosineDistance(v,w):

    unit = lambda x: x / np.sqrt(np.dot(x,x))
    if (v**2).sum() < 1e-7:
        raise Exception('First argument to cosineDistance is close to zero vector')
    if (w**2).sum() < 1e-7:
        raise Exception('Second argument to cosineDistance is close to zero vector')    
    return np.dot( unit(v), unit(w) )


# In[ ]:


# Training Params
epochs = 1000
batch_size = 100

# Hyperparams
embedding_dim = 128
neg_sample_size = 64 #300
learning_rate = 0.051
# momentum = 0.5

# word2vec and validation
radius = 15#2       # sample window radius
repeat_num = 3#2   # number of times to sample each word
head_subset_size = 100 # sample only from head of distribution for monitoring progress
validation_sample_size = 6


# Init NN
word_id_counts = Counter( { word2int[w] : words[w] for w in words } )

net = littleNN(word_id_counts, embedding_dim, neg_sample_size, learning_rate)

def trainNN(word_id_counts):

    
    # Data iterator
    moreData = iterData(settings["data_path"], batch_size, jump_around=True) # data iterator
    

    ###############
    # Random sample from dictionary to observe closest cosine distances
    head_of_distribution = [ id_count[0] for id_count in word_id_counts.most_common( head_subset_size )]
    samp_ids = random.sample(head_of_distribution, validation_sample_size)
    pS = pd.Series([w_id for w_id in word_id_counts.keys() if w_id not in samp_ids])
        
    ##############
    #
    prev_loss = 1000
    
    for e in range(epochs):
        
        # for v in [net.word2vec(i) for i in samp_ids]:
        #     print(v)
        # print("-------------------\n")
        
        
        ## View sampled words' neighbors every epoch
        if e % 200 == 0:
            for samp_id in samp_ids:
                neighbor_ids = pS.apply( lambda w_id: cosineDistance(net.word2vec(samp_id), net.word2vec(w_id)) ).nlargest(6).index.values
                print("Words close to {}: {}".format(int2word[samp_id], [int2word[i] for i in neighbor_ids] ) )
            print("------------\n")
        

        ## Train ##
        
        batch = next(moreData)
        batch = [ word2int[word] for word in batch if word2int.get(word,None) is not None ]
        
        word_id_batch, target_batch = makeInputsTargets(batch, radius=radius, repeat_num=repeat_num )

        a0, z1 = net.forwardPass(word_id_batch)

        # Report loss Early break tweak learning rate
        if e % 20 == 0:
            loss = net.crossEntropy(net.Softmax(z1), target_batch)
            print("Loss: ", loss)
            if loss - prev_loss > 2:
                print("Seems to be diverging")
                break
            if loss - prev_loss > 0.1:
                net.lr *= 0.1
            prev_loss = loss
            
        
        net.sampledBackProp(a0, z1, word_id_batch, target_batch)

        
## Run training

trainNN(word_id_counts)


# In[ ]:


samp_id = word2int["three"]

pS = pd.Series([w_id for w_id in word_id_counts.keys()])
neighbor_ids = pS.apply( lambda w_id: cosineDistance(net.word2vec(samp_id), net.word2vec(w_id)) ).nlargest(6).index.values

print("Words close to {}: {}".format(int2word[samp_id], [int2word[i] for i in neighbor_ids] ) )
print("------------\n")


# In[ ]:


a = np.random.randint(0,10, (1,10))


# In[ ]:


pS = pd.Series(np.array([1,2,3,4,5,74,7,78]))
pS.apply(lambda x: 1/x).nsmallest(3).index.values

