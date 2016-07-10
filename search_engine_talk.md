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

Considering each document in isolation doesn't keep track of the global importance of a word. For example, "the" appears in both documents and it may seem important because the relative frequency is so high. But at a global level, since it appears at relatively frequency in both documents, we may consider it less important. To mathematically encode this global frequency information, we move to the `Term Frequency - Inverse Document Frequency (TF-IDF)` representation. This requires constructing a vocabulary.

Constructing a vocabulary may or may not be an expensive operation. Another vectorization which uses the so-called `Hashing Trick` does not require construction of a vocabulary.

### Another Vectorization: Hashing Trick vectors
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


### Another Vectorization: word2vec

## Closeness in Vectorization: Cosine similarity


## The Basic Search Engine
### The document model

### Boolean Query

### Inverted Index Data Structure

## The Elasticsearch (Lucene) document model

### What about JOINS?


## Vector closeness and Elasticsearch "Relevance" score

## Clustering, a classic machine learning problem
### k-means

### Expectation maximization and Gaussian Mixture as a smooth k-means

### Topic Modeling, a form of "soft clustering"

## Anomaly Detection, a variant of Clustering
