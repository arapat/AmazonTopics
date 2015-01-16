This is an on-going project on analyzing online text datasets.

# Data
There are two different datasets being analyzed.

* Amazon reviews
	* It can be downloaded from http://snap.stanford.edu/data/web-Amazon.html.
* Some distributional similarity experiments
	* It can be downloaded from http://www.cs.cornell.edu/home/llee/data/sim.html

# Codes
We are using [Apache Spark](https://spark.apache.org/) and [IPython Notebook](http://ipython.org/notebook.html) in this work. Most of the computational parts, such as word counts, divergence calculation, is working on Spark. Then we use IPython Notebook to visualize the result, such as the distance between the vectors.

IPython Notebook files are in `notebook/`.

Under `python/`, there are some helper scripts, such as data parsers.

Spark codes are under `spark/`. 

* `spark/DistributionalSimilarity`: The data source is from distributional similarity experiments. Constructions vector for every noun. The elements of the vectors are the number of times that each verbs occur with the corresponding noun. The output consists of the Jensen-Shannon divergences between each pairs of the vectors, the vectors, and the mapping between the indices of the vectors and the corresponding words.
* `spark/JSDivergence`: The output is same as `spark/DistributionalSimilarity`, with the data source is amazon reviews.
* `spark/KLDivergence`: Compute the Kullback–Leibler divergence from each vectors to the overall word distribution without constructing the vectors explicitly.
* `spark/NearestNeighbor`: Assign every word in the corpus to the closest center.

