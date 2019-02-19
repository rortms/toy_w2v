
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[7]:


# Infinite data iterator
def iterData(filename, batch_size):

    # Iterator reading file in batches
    def get_batch():
    
        with open(filename, "r") as datfile:

            position, offset = 0, batch_size

            while True:
                 yield datfile.read(batch_size)
                 position = datfile.seek(position+batch_size)

    batch = get_batch()

    # When file is exhausted, start over with new get_batch ierator
    while True:
        b = next(batch)
        if b:
            yield  b
        else:
            batch = get_batch()
        


# In[8]:


# Test infinite iterator
fn = "./testf"

with open(fn, "r") as tf:
    lines = tf.read()
print(len(lines))

# Only 54 charactors reused indefinitely
testIter = iterData(fn, 5)
print(list( (next(testIter) for k in range(100)) ) )


# In[ ]:


with open("./config", "r") as conf:
    data_file = conf.read().split(1)

moreData = iterData(data_file, 1000)

