# Text Representation, Search Engines, and Basic Clustering on Documents

## Purpose
- Since one of the datastores we use is a document-based search engine called `Elasticsearch`, it'll be good to know how it roughly works.
- We'll cover representation of text documents, basic search engine theory, and possible first steps in clustering on text

## Text Representation
### The Bag of Words model
Suppose we have a text string `doc1 = "The dog chases the cat and the cat runs from the dog"`

The most basic way we can represent this logically to the computer is to consider it as a set of strings `{"The", "dog", "chases", "the", "cat", "and", "runs", "from" }`.

Perhaps we want to keep track of word counts. See for example that "cat" and "dog" appear twice, but "and" only once. So we have

    {"The": 1,
    "dog": 2,
    "chases": 1,
    "the": 3,
    "cat": 2,
    "and": 1,
    "runs": 1,
    "from": 1}

Notice we have destroyed the information on the order of words. So in this representation, the text string `"The cat chases the dog and the dog runs from the cat"` would have the same representation.

If you have a second `doc2 = "the dog barks"`, we would need to update the representation of `doc1` to include a key for "barks", with count 0, to make:

    {"The": 1,
    "dog": 2,
    "chases": 1,
    "the": 3,
    "cat": 2,
    "and": 1,
    "runs": 1,
    "from": 1,
    "barks": 0}

Now, you may argue that the mapping `{"barks": 0}` was implicit in the previous representation and unnecessary. That's true. Each representation can be `sparse`, which means it only holds "non-zero" information. This works best when almost all information is "zero".

This is the `Bag of Words` representation of text.


### Term Frequency Vectorization of Bag of Words

If we take into account the length of the document we can normalize the documents, taking only into account the relative frequency of words in a document. The `length` of `doc1` is 12. Dividing each count by 12, we get

    {"The": 1/12,
    "dog": 2/12,
    "chases": 1/12,
    "the": 3/12,
    "cat": 2/12,
    "and": 1/12,
    "runs": 1/12,
    "from": 1/12}

In particular, the sum of these frequencies sums to 1.

In particular, in this normalization, the document `doc1` and

    doc1 + " " + doc1 =
    "The dog chases the cat and the cat runs from the dog
    The dog chases the cat and the cat runs from the dog"

have the same representation.

Consider the document `doc2 = "the dog barks"` again. Note that for the bunch of documents `doc1, doc2` we did not need to construct a `vocabulary`, a set of all words in all documents in the bunch. We considered each document alone to build this representation. As a mathematical vector, we can consider it a `sparse vector`, where we only hold "non-zero" information.

### Term Frequency - Inverse Document Frequency (TF-IDF): Term Frequency Vectorization with Global Rarity Scaling

Considering each document in isolation doesn't keep track of the global importance of a word. For example, "the" appears in both documents and it may seem important because the relative frequency is so high. But at a global level, since it appears at high relatively frequency in both documents, we may consider it less important. To mathematically encode this global frequency information, we move to the `Term Frequency - Inverse Document Frequency (TF-IDF)` representation. This requires constructing a vocabulary and inverse document mapping.

Constructing a vocabulary may or may not be an expensive operation. Another vectorization which uses the so-called `Hashing Trick` does not require construction of a vocabulary.

Each vector in `TF-IDF` representation is normalized by length and consists of features, for each word w, `tf_idf(w) = term frequency of w * inverse document frequency of w = tf(w) * idf(w)`. The term frequency is computed above. The inverse document frequency `idf(w)` is given by `log ((total number of documents) / (1 + number of documents containing w))`. Note that this is a number global over an entire bunch of documents. The `1+` is just so you don't divide by 0, and note this number doesn't make sense if you have no documents.

#### Exercise
Take ten text documents and compute the `idf(w)` for each word w. Then compute the vector representations of each document.

#### Exercise
Think about how you would handle text for which there are no spaces or delimiters. This is the first sentence from Japanese Wikipedia's article on Japan.

日本国（にっぽんこく、にほんこく）、または日本（にっぽん、にほん）は、東アジアに位置する日本列島（北海道・本州・四国・九州の主要四島およびそれに付随する島々）及び、南西諸島・小笠原諸島などの諸島嶼から成る島国である[1]。日本語が事実上の公用語として使用されている。首都は事実上東京都とされている。

### (Optional) Another Vectorization: Hashing Trick vectors
Instead of mapping a document to its word frequencies, we can map it to a sparse `bitset` in a sufficiently large space. This method bypasses the need to construct the vocabulary, and can handle what would have amounted to a large unknown and frequently changing vocabulary.

Take `b=24` for example. We can map a document to a `2 ** b = 2 ** 24` length sparse bitset. Why such a large space? The space should be large enough to capture all the interactions between the words in the document while still giving a flat data structure.

The `Hashing Trick` makes use of a hash function to determine the representation.

    def toHashVector(String doc, int b):
        termFrequencyVector = term frequency vector of doc

        resultLength = 2 ** b
        result = new Bitset of all 0 of length resultLength

        for each nonzero entry (word, freq) in termFrequencyVector:
            idx = hash((word, freq)) mod resultLength
            result.flip(idx)

        return result

Note that the features are not interpretable, whereas the term frequency vector features are exactly the word frequencies of each word.

#### Exercise
Use the language of your choice. Find a sparse(!) bit vector implementation and implement this hashing trick. Do NOT instantiate a boolean array of length 2 ** 24.

#### Exercise
Assume your hash function meets the Uniform Hashing Assumption (i.e., roughly all possible output from your input is uniformly distributed under the hash function). Does the hash vector also meet this a similar uniform distribution assumption?

### (Optional) Another Vectorization: word2vec
Another representation of word documents, called `word2vec` maps documents to a space where arithmetic `(+, -)` "makes sense". Roughly speaking, we fix a small integer length such as `300` and we map documents to vectors of real numbers of length 300. This representation is achieved by doing deep learning on documents, so the machine can find latent relationships between words in documents. The features in the vector are not interpretable.

This representation can track latent relationships between words, and the notion of closeness, as calculated below, can be much finer.

#### Exercise
Look at the papers and/or source code for the original implementation out of Google
https://code.google.com/archive/p/word2vec/ .
If you know Apache Spark or Python, try out a demo for an implementation in Spark MLlib or gensim, both open source.

## Closeness in Vectorization: Cosine similarity

Consider two unit length vectors `u = (ux, uy)`, `v=(vx, vy)` in xy-coordinates. Then the formula holds `cos(t) = u . v`, where `.` is the dot product. In this case, `u . v = ux * vx + uy * vy`. The angle t is set between 0 and 90 degrees. Two documents are identical if t is 0, and totally dissimilar if t is 90. We set the `cosine similarity sim(u, v) = cos(t) = u . v`. This number is between 0 and 1 and has the correspondence of `0 degrees: similarity 1` and `90 degrees: similarity 0`.

Let's we consider for example that the x-axis is word frequency of "cat", and the y-axis is for "dog". We take several documents:

    doc1 = "cat dog"
    doc2 = "dog cat"
    doc3 = "dog"
    doc4 = "cat"

Let's abuse notation and we find that `sim(doc1, doc2) = 1` because the words are the same, and that `sim(doc3, doc4) = 0` because they share no words.

#### Exercise
Implement cosine similarity on your favorite vectorization.

## The Basic Search Engine
### The document model
For simplicity, a (text) `document` is uniquely identified by a string called `doc_id` and is a sparse TF-IDF representation of the text.

#### Exercise
How is this different from a relational model? How would you store the hashing trick or word2vec representations under the document model?

### Boolean Query
We focus on queries that return sets of document ids containing or not containing certain words. Set operations can be decomposed into one of three (or two) operations:

    and(word1, word2) = all docs with word1 and word2
    or(word1, word2) = all docs with word1 or word2
    withAndNot(includedWord, excludedWord) =
        all docs with includedWord but without excludedWord

#### Exercise
Given `or(word1, word2)` and `withAndNot(includedWord, excludedWord)`, implement `and(word1, word2)`.

### Inverted Index Data Structure

We want a data structure that will support fast lookups by word of all documents containing that word and set operations intersection, union, and negation.

One data structure we can use for this is an `inverted index`.

We have a key value data structure. Keys are words. Values are sets of all documents containing the word. Intersection, Union, and Negation can be done in time proportional to size of the sets. Lookups by word are constant or log time.

#### Exercise
Implement an inverted index with the following interface.

    InvertedIndex
        InvertedIndex(list of documents)
        boolean isInIndex(word)
        Set of doc ids and(word1, word2)
        Set of doc ids or(word1, word2)
        Set of doc ids withAndNot(includedWord, excludedWord)



## The Elasticsearch (Lucene) document model
A key-value data structure where keys are strings called `fields` and values are strings, booleans, numbers, another document or arrays of such. When a document value is a string, you may choose how you want to process the text. It is usually processed at a TF-IDF vector as above.

The document is one of several packaged in an `index`, which is analogous to databases in relational databases. There is a `mapping`, which holds schema information. Documents need not conform to this schema, but if they don't their values may not be stored properly.

## Elasticsearch "Relevance" score
Given a query, what does it mean for a document to be "relevant" or "more relevant" or "less relevant"?

Suppose we have a single word query: "CONTAINS word". One way is to rely on the `TF-IDF(word, doc)` for each word in the query. Recall that this is just the product `term freq of word in doc * inverse document freq of word`. So the higher it is for a fixed word but between two documents, you choose the document with the higher one. This make sense because the document with the higher number , the word is more frequent in it.

If you have several words in your query, you can add the TF-IDF scores for a fixed document across all the words.

What if your query is "CONTAINS word1 AND NOT CONTAINS word2"? Simply filter the document containing word2 out of your result set, no matter how relevant.

It turns out, this is pretty close to simply asking, assuming the query is a document, for the most similar documents, in terms of cosine similarity.

#### Exercise
Verify this above claim.

## Clustering, a classic machine learning problem
Suppose you have many documents and you have settled on a vectorization that is appropriate. Recall that Elasticsearch by default uses TF-IDF vectorization.

In our case, we may have documents containing information on people. Each document contains date of birth, name, and favorite books. For simplicity, we can take each document as a giant text string. For example, we have `doc1 = "1990-01-01 warshmellow 'I am a cat' 'Ten Nights of Dreams'"` and `doc2 = "1975-01-01 chocolatebunny 'Fun Home' 'The Odyssey'""`.

Let's say we have 1000 such. We want to see if they form natural clusters. Maybe 10 natural clusters. This is called a `clustering` problem. Another name for this problem is called `market segmentation`. Another name is `unsupervised learning in machine learning`.

### k-means
The simplest form of clustering has you fix `k` number of clusters you think there will be, then calculating cosine similarities between vectors. There will be `k` clusters in the end, hopefully all the points in a cluster will have very high cosine similarities, and points in two different clusters will have low cosine similarities. You try different `k` to optimize for this simultaneous condition.

### Expectation maximization and Gaussian Mixture as a smooth k-means
For a variety of mathematical reasons, `k-means` sometimes gives you bad results because it is too "rigid". Roughly speaking, it imagines all clusters to be non-overlapping "circles", and you can split hairs over which circle a point belongs to.

A "smooth" version of k-means allows clusters to be ellipses and allows points to belong to different ellipses with varying degree of belonging.

Framing a problem this way is called a `Gaussian mixture` problem: it assumes there are `k` normal distributions that all points belong to with a certain probability.

The `Expectation maximization` algorithm is used to compute these probabilities.

### Topic Modeling, a form of "soft clustering"

The above methods are somewhat "hard" clustering because they explicitly use similarity to compute shapes and boundaries.

Another way is "soft clustering", which uses "latent" or hidden information to compute clusters.

If you assume all vectors are linear vectors, then you can extract linear "latent" information using matrix methods.

## Anomaly Detection, a variant of Clustering
