"""
Word count app.
"""
from operator import add
from pyspark.sql import SparkSession

sparkSession = SparkSession.builder.master("local[*]").appName("wordcount").getOrCreate()
text = sparkSession.read.text("LICENSE").rdd.cache()

lines = text.map(lambda r: r[0])
counts = lines.flatMap(lambda l: l.split(' ')).map(lambda w: (w, 1)).reduceByKey(add)
words = counts.collect()
for (word, count) in words:
    print("%s: %i" % (word, count))

sparkSession.stop()
