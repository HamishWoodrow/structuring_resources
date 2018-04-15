from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.utils.validation import check_is_fitted
import urllib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
import nltk
import pickle
import time
import csv
import numpy as np
from collections import defaultdict
import array
import scipy.sparse as sp
import numbers
from operator import itemgetter
from tqdm import tqdm
import sys
import pathlib
import os
import glob
import requests

def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


class concept_vectorizer(BaseEstimator):
    """A modified count vectorizer method based on Sklearn
    
    The count vectorizer is modified to create and look for bigrams concepts in a text
    These method parses the corpus of documents twice.  Initially to extract a bigram 
    count vector for each document (collocation is used to extend possible bigrams).
    Then once the matrix of possible concepts is created, a local instance of DBPEDIA
    is queired to verify if the bigram is a valid concept.  In addition, DBPEDIA
    provides disambiguation terms (i.e. variations for the bigram with the same route
    meaning).
    This creates a reduced number of dimensions, by removing n_grams not existing in
    DBPEDIA.  This new list of n_grams is then used as the fixed vocabulary to search 
    for within the corpus.  Creating a new count vectorize matrix of dimensions : 
    [n_documents, n_fixed_concepts].
    A TFIDF transform is then performed on this matrix.
    
    Modifications
    -------------
    
    """
    def __init__(self, input='content', encoding='utf-8',
             stop_words=None,max_df=1.0, min_df=1,max_features=None,
                dtype=np.int64, binary=False,fixed_vocab=False,path=True,
                doc_path_type='db_cursor'):
        self.input = input
        self.encoding = encoding
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.dtype = dtype
        self.binary = binary
        self.fixed_vocab = fixed_vocab
        self.path = path
        self.doc_path_type = doc_path_type
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.vocabulary = None

    def tokenizer_bigram(self, text_data):
        """Tokenizes a text into bigrams, trigrams and quadgrams, with collocation
        
        If no fixed vocabulary list is given the tokenizer is a simple bigram
        tokenizer with collocation.  Allowing for n_gram tuples to be re-ordered.
        
        If a fixed vocabulary (dictionary) is given, the function will tokenize the 
        text into bigrams, trigrams and quadgrams with collocation.  Then each n_gram 
        token will be looked up in the fixed_vocab dictionary and the frequency counted.
        
        Parameters
        ----------
        text_data : string
            A string of the documents text to be tokenized.
            
        Attributes
        ----------
        filt_term_freq : dict
            Containg n_gram frequencies for a document.  Key is the token
            value is frequency of token.
        """
        tokens = nltk.word_tokenize(text_data,language='english')
        token_lst = []
        for token in tokens:
            token = re.findall('[A-Za-z -/\']*',token)
            if (len(token) > 1):
                if (token[0].lower()) not in self.stop_words:
                    token_lst.append(token[0].lower())
        if not self.fixed_vocab:
            bigram_measures = nltk.collocations.BigramAssocMeasures()

            finder=nltk.collocations.BigramCollocationFinder.from_words(token_lst)

            likely_term = set(finder.nbest(bigram_measures.likelihood_ratio, 200))
            finder.apply_word_filter(lambda w: len(w) < 3)
            term_freq = dict(finder.ngram_fd)
            filt_term_freq = {}
            for term, freq in term_freq.items():
                if term in likely_term:
                    filt_term_freq[term] = freq

        elif self.fixed_vocab:
            finder2 = nltk.collocations.BigramCollocationFinder.from_words(token_lst, window_size=5)
            finder3 = nltk.collocations.TrigramCollocationFinder.from_words(token_lst, window_size=5)
            finder4 = nltk.collocations.QuadgramCollocationFinder.from_words(token_lst, window_size=5)

            freq_2 = finder2.ngram_fd
            freq_3 = finder3.ngram_fd
            freq_4 = finder4.ngram_fd
            freq_dict = defaultdict(int)

            concepts = self.fixed_vocab

            for con, key_con in concepts.items():
                if freq_2[con] > 0:
                    freq_dict[key_con] += freq_2[con]
                elif freq_3[con] > 0:
                    freq_dict[key_con] = freq_3[con]
                if freq_4[con] > 0:
                    freq_dict[key_con] = freq_4[con]

            filt_term_freq = freq_dict

        return filt_term_freq

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in xrange(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't contain index "
                               "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def _alliter(self,p):
        yield p
        for sub in p.iterdir():
            if sub.is_dir():
                yield from self._alliter(sub)
            else:
                yield sub

    def _count_vocab_db(self, cursor, fixed_vocab):
        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__

        j_indices = []
        doc_indexer = defaultdict(list)
        indptr = array.array(str("i"))
        values = array.array(str("i"))
        indptr.append(0)
        
        doc_idx = 0
        total_docs = cursor.count()
        pbar = tqdm(total=total_docs)
        while cursor.alive:
            doc = cursor.next()
            if doc['abstract'].get('p'):
                abstract = doc['abstract']['p']
                feature_counter = {}
                try:
                    token_terms = self.tokenizer_bigram(abstract)
                    for feature, freq in token_terms.items():
                        feature_idx = vocabulary[feature]
                        feature_counter[feature_idx] = freq
                    j_indices.extend(feature_counter.keys())

                    values.extend(feature_counter.values())
                    indptr.append(len(j_indices))
                    doc_indexer[doc_idx] = doc['_id']
                    doc_idx+=1
                    pbar.update(1)
                except:
                    pbar.update(1)

        self.doc_indexer = doc_indexer
        pbar.close()

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                         shape=(len(indptr) - 1, len(vocabulary)),
                         dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X
                
                
    def _count_vocab(self, path, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""

        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__

        j_indices = []
        doc_indexer = defaultdict(list)
        indptr = array.array(str("i"))
        values = array.array(str("i"))
        indptr.append(0)

        tot_files = len(list(pathlib.Path(path).glob('**/*')))
        file_g = self._alliter(pathlib.Path(path))
        doc_idx = 0
        for i in tqdm(range(tot_files)):
            fpath = next(file_g)
            if fpath.name[-3:] == 'txt':
                with open(fpath) as fhand:
                    doc = fhand.read().replace('\n', '')
                feature_counter = {}
                token_terms = self.tokenizer_bigram(doc)
                for feature, freq in token_terms.items():
                    feature_idx = vocabulary[feature]
                    feature_counter[feature_idx] = freq
                j_indices.extend(feature_counter.keys())

                values.extend(feature_counter.values())
                indptr.append(len(j_indices))
                doc_indexer[doc_idx] = fpath
                doc_idx+=1

        self.doc_indexer = doc_indexer

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                         shape=(len(indptr) - 1, len(vocabulary)),
                         dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X

    def fit_transform(self, path, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        #if isinstance(raw_documents, six.string_types):
            #raise ValueError(
                #"Iterable over raw text documents expected, "
                #"string object received.")

        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features
        
        if self.doc_path_type == 'db_cursor':
            cursor = path
            vocabulary, X = self._count_vocab_db(cursor,
                                          self.fixed_vocabulary_)
        else:
            vocabulary, X = self._count_vocab(path,
                                          self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)

            self.vocabulary_ = vocabulary

        return X

    def _sort_features(self, X, vocabulary):
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(six.iteritems(vocabulary))
        map_index = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def _limit_features(self, X, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(six.iteritems(vocabulary)):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.
        Parameters
        ----------
        X : {array, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        X_inv : list of arrays, len = n_samples
            List of arrays of terms.
        """
        self._check_vocabulary()

        if sp.issparse(X):
            # We need CSR format for fast row manipulations.
            X = X.tocsr()
        else:
            # We need to convert X to a matrix, so that the indexing
            # returns 2D objects
            X = np.asmatrix(X)
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocabulary = terms[np.argsort(indices)]

        return [inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)]

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        #self._check_vocabulary()

        return [t for t, i in sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))]
    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fit-ed)"""
        msg = "%(name)s - Vocabulary wasn't fitted."
        check_is_fitted(self, 'vocabulary_', msg=msg),

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)

def concept_validator(counter):
    """Validate a token vs DBPEDIA for existance of the term as a concept
    
    The feature names for a vector are taken and each token is queried for existance
    in a local instance of DBPEDIA.  
    If a result is returned the top matching results data is used for the new label
    for this token and this token is added to a concept dictionary storing all
    validated concepts.
    
    Parameters
    ----------
    counter : object
        The count_vecotorized function to get the feature names.
    
    Attributes
    ----------
    concept_dict : dict
        Validated concept label with values of the  URI address for the term in DBPEDIA
        and the original token queried.
    """
    concept_dict = {}
    for pos_concept in tqdm(counter.get_feature_names()):

        prefix = 'http://localhost:1111/api/search/KeywordSearch?QueryClass=&QueryString='
        query = pos_concept[0] + '_' + pos_concept[1]
        url = prefix + query

        page = urllib.request.urlopen(url).read()
        root = ET.fromstring(page)

        if root.findall('.//{http://lookup.dbpedia.org/}Result'):
            concept_label = root.findall('.//{http://lookup.dbpedia.org/}Label')[0].text
            URI = root.findall('.//{http://lookup.dbpedia.org/}URI')[0].text
            concept_dict[concept_label] = [pos_concept, URI]
    return concept_dict

def dbpedia_metadata(concept_dict):
    """Gather metadata for terms in validated concept dictionary.
    
    Based on a dictionary with keys of terms validated to exist in DBPEDIA, query
    DBPEDIA to get the metadata for this term in order to use for ontology 
    creation and also for disambiguation.  The redirects for the terms are used
    for the disambiguation and subject used for the parent node.  The abstract
    for each term is also logged.
    
    Parameters
    ----------
    concept_dict : dict
        Validated dictionary of terms with values including the URI for DBPEDIA
        reference for the term.
    
    Attributes
    ----------
    vocab_metadata_dict : dict
        Term key with values giving, subject and redirect nodes along with original
        term queried and abstract for token.
    """
    
    vocab_metadata_dict = defaultdict()
    error_count = 0
    count=0
    for concept,dbpedia_link in tqdm(concept_dict.items()):
        num_terms = len(concept.split())
        if (num_terms <= 5) and (num_terms != 1):
            try:
                prefix = dbpedia_link[1].replace('resource','data')
                url = prefix + '.n3'
                response = urllib.request.urlopen(url).read().decode()

                subject = re.findall('dbc:([\w]+)',response)
                redirects = re.findall('dbr:([\w]+)\tdbo:wikiPageRedirects',response)
                abstract = re.findall('\"(.*)\"@en',response)
                idx_max_abstract = np.argmax([len(a) for a in abstract])
                abstract = abstract[idx_max_abstract]
                bigram = tuple(dbpedia_link[0])
                # Create dictionary
                vocab_metadata_dict[concept] = [url,subject,redirects,bigram,abstract]
            except:
                with open('dbpedia_error_log.txt','a') as fb:
                    fb.write(concept)
                error_count += 1
            count+=1
            if count%100 == 0:
                print('Current count: ', count)

    print('Total error count:',error_count)
    return vocab_metadata_dict

def fixed_concept_dict(vocab_dict):
    """Production of fixed terms dictionary based on dbpedia extraction for disambiguation
    
    Takes the vocab_metadata_dict created by dbpedia_metadata function and
    transforms this into a dictionray with the key being the primary concept_label
    from dbpedia and the values being disambiguation terms (based on the redirects
    and also the original query token).
    
    Parameters
    ----------
    vocab_metadata_dict : dict
        Dictionary containing concept_label and for each a list of metadata from 
        dbpedia.
        
    Attributes
    ----------
    fixed_concept_dict : dict
        Keyed by primary concept label with values being all possible terms directly
        related to primary concept label.  Creating a disambiguation dictionary for
        like terms.
    
    """
    
    fixed_concept_dict = {}
    for key, metadata in tqdm(vocab_dict.items()):
        key_gram = tuple(key.lower().split(' '))
        concept_label = key.replace(' ','_')
        fixed_concept_dict[key_gram] = concept_label
        orig_bigram = tuple(metadata[3])
        #print(orig_bigram)
        #print(concpet_label)
        fixed_concept_dict[orig_bigram] = concept_label
        for each in metadata[2]:
            each = tuple(each.lower().split('_'))
            fixed_concept_dict[each] = concept_label
    return fixed_concept_dict