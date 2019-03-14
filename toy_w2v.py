
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
                    offset = np.random.randint(1,4) * batch_size // 3
                    
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


# In[6]:


# Construct vocabulary set
all_words = Counter(next(moreData))
print(len(all_words))
for k in range(70000):
    all_words.update(next(moreData))
print(len(all_words))
        


# In[7]:


words = Counter( { w[0] : w[1] for w in all_words.most_common(int(80e3)) } )
words['UNKN'] = 0
for w in all_words:
    if words.get(w, None) is None:
        words['UNKN'] += 1


# In[8]:


print(len(words))


# In[9]:


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


######################################
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


# In[11]:


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


# In[12]:


np.array([[1,2,4],[1,1,1]]) *  np.array([[1,0,1],[0,0,2]])


# In[13]:


csr_matrix(2*np.random.randint(0,2, (6,4)).T) @ (4*np.eye(6))


# In[15]:


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


# In[16]:


######################
# Explanatory Ex 2

l = 7
ints = np.array([range(11, 3, -1) for k in range(l)])
indices = np.array([[5,0] for k in range(l)])
print(ints)
print(indices)
select_ints = ints[np.arange(l)[:,None],indices]

print(select_ints)


# In[17]:


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


# In[18]:


help(csr_matrix.dot)


# In[19]:


def cosineDistance(v,w):

    unit = lambda x: x / np.sqrt(np.dot(x,x))
    if (v**2).sum() < 1e-7:
        raise Exception('First argument to cosineDistance is close to zero vector')
    if (w**2).sum() < 1e-7:
        raise Exception('Second argument to cosineDistance is close to zero vector')    
    return np.dot( unit(v), unit(w) )


# In[20]:


# Training Params
epochs = 1000
batch_size = 100
# Hyperparams
embedding_dim = 128
neg_sample_size = 64 #300
learning_rate = 0.05
# momentum = 0.5

# word2vec and validation
radius = 4#2       # sample window radius
repeat_num = 3#2   # number of times to sample each word
head_subset_size = 100 # sample only from head of distribution for monitoring progress
validation_sample_size = 6


# In[21]:


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

    def report_closest():
        for samp_id in samp_ids:
            neighbor_ids = pS.apply( lambda w_id: cosineDistance(net.word2vec(samp_id), net.word2vec(w_id)) ).nlargest(6).index.values
            print("Words close to {}: {}".format(int2word[samp_id], [int2word[i] for i in neighbor_ids] ) )
        print("------------\n")        
    ##############
    #
    prev_loss = 1000
    
    for e in range(epochs):
        
        # for v in [net.word2vec(i) for i in samp_ids]:
        #     print(v)
        # print("-------------------\n")
        
        
        ## View sampled words' neighbors every epoch
        if e % 200 == 0:
            report_closest()
        
        ## Train ##
        
        batch = next(moreData)
        batch = [ word2int[word] if word2int.get(word,None) is not None else word2int['UNKN'] for word in batch  ]
        
        word_id_batch, target_batch = makeInputsTargets(batch, radius=radius, repeat_num=repeat_num )

        a0, z1 = net.forwardPass(word_id_batch)

        # Report loss Early break tweak learning rate
        if e % 20 == 0:
            loss = net.crossEntropy(net.Softmax(z1), target_batch)
            print("Loss: ", loss)
            if loss - prev_loss > 2:
                print("Seems to be diverging")
                report_closest()
                break
            if loss - prev_loss > 0.1:
                net.lr *= 0.1
            prev_loss = loss
            
        
        net.sampledBackProp(a0, z1, word_id_batch, target_batch)
        
    report_closest()
    
## Run training

trainNN(word_id_counts)


# In[22]:


import tensorflow as tf
tf.reset_default_graph()

# Training Params
epochs = 1000
batch_size = 1000
# Hyperparams
embedding_dim = 128
neg_sample_size = 64 #300
learning_rate = 0.05

# word2vec and validation
radius = 6#2       # sample window radius
repeat_num = 3#2   # number of times to sample each word
head_subset_size = 100 # sample only from head of distribution for monitoring progress
validation_sample_size = 6



###############
moreData = iterData(settings["data_path"], batch_size)

# Input
word_indices = tf.placeholder(tf.int32, [None])
labels = tf.placeholder(tf.int32, [None, None])

# Embedding
embedding_layer = tf.get_variable("Embedding_Layer",
                                  initializer=tf.truncated_normal([len(words), embedding_dim]))

embed_vecs = tf.nn.embedding_lookup(embedding_layer, word_indices)

softmax_out = tf.get_variable("Softmax_Out", 
                              initializer=tf.truncated_normal([len(words), embedding_dim])) # Order reversed from expected

biases_out = tf.get_variable("Biases_Out",
                             initializer=tf.zeros(len(words)))

#
loss = tf.reduce_mean ( tf.nn.sampled_softmax_loss( softmax_out,
                                                    biases_out,
                                                    labels,
                                                    embed_vecs,
                                                    neg_sample_size,
                                                    len(words) ) )
#
optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

with tf.Session() as sesh:
    
    sesh.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = next(moreData)
        batch = [ word2int[word] if word2int.get(word,None) is not None else word2int['UNKN'] for word in batch  ]        
        word_ids, label_ids = makeInputsTargets(batch, radius, repeat_num)

        d = { word_indices : word_ids, labels : np.array(label_ids)[:,None] }
        
        _, l = sesh.run([optimizer, loss],  feed_dict=d)
        
        if i % 500 == 0:
            print(l)


# In[ ]:


help(tf.nn.sampled_softmax_loss)


# In[ ]:


samp_id = word2int["three"]

pS = pd.Series([w_id for w_id in word_id_counts.keys()])
neighbor_ids = pS.apply( lambda w_id: cosineDistance(net.word2vec(samp_id), net.word2vec(w_id)) ).nlargest(6).index.values

print("Words close to {}: {}".format(int2word[samp_id], [int2word[i] for i in neighbor_ids] ) )
print("------------\n")


# In[ ]:


pS = pd.Series(np.array([1,2,3,4,5,74,7,78]))
pS.apply(lambda x: 1/x).nsmallest(3).index.values

