# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Top 50 palabras de El Quijote

# COMMAND ----------

import urllib

with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/don-quijote.txt.gz') as response:
  gzipcontent = response.read()

with open("/tmp/don-quijote.txt.gz", 'wb') as f:
  f.write(gzipcontent)

dbutils.fs.cp("file:/tmp/don-quijote.txt.gz",'/tmp/')

# COMMAND ----------

l1 = [1,2,3]
print(l1)
'a b c'.split(' ')

# COMMAND ----------

with urllib.request.urlopen('https://raw.githubusercontent.com/stopwords-iso/stopwords-es/master/stopwords-es.txt') as response:
  gzipcontent = response.read()

with open("/tmp/stopwords-es.txt", 'wb') as f:
  f.write(gzipcontent)
  
sws_f = []
with open("/tmp/stopwords-es.txt", 'r') as f:
  sws_f = f.readlines()
sws_f1 = [sw.rstrip() for sw in sws_f]

# COMMAND ----------

len(sws_f1)

# COMMAND ----------

docs = sc.textFile("file:/tmp/don-quijote.txt.gz")

lower = docs.map(lambda line: line.lower())
# print(lower)
# print(type(lower.take(10)))
# print(lower.take(10))

words = lower.flatMap(lambda line: line.split(' '))
# print(words.take(10))

def filter_func(w):
  return w not in sws_f1

# w_nsws = words.filter(lambda w: w not in sws_f1)

w_nsws = words.filter(filter_func)

counts = w_nsws.map(lambda word: (word, 1))

# counts = words.map(lambda word: (word, 1))
# print(counts)
# print(counts.take(10))

from operator import add
freq = counts.reduceByKey(add)
# print(freq.take(10))

invFreq = freq.map(lambda t: (t[1],t[0]))
print(invFreq.top(50))


# COMMAND ----------

add?

# COMMAND ----------

test = sc.parallelize([1,2,3,4])
print(test)
test.filter(lambda e: e < 4).collect()