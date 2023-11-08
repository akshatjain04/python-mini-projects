import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plag

def test_similarity_identical_docs():
    doc1 = "This is a test document"
    doc2 = "This is a test document"
    result = plag.similarity(doc1, doc2)
    assert np.isclose(result, 1), "Test failed: For identical documents, similarity score should be 1"

def test_similarity_no_common_words():
    doc1 = "This is a test document"
    doc2 = "Apple banana mango"
    result = plag.similarity(doc1, doc2)
    assert np.isclose(result, 0), "Test failed: For documents with no common words, similarity score should be 0"

def test_similarity_partial():
    doc1 = "This is a test document"
    doc2 = "This is a sample document"
    result = plag.similarity(doc1, doc2)
    assert 0<result<1, "Test failed: For documents with partial similarity, similarity score should be between 0 to 1"

def test_similarity_empty_docs():
    doc1 = ""
    doc2 = ""
    result = plag.similarity(doc1, doc2)
    assert result in [0, 1], "Test failed: For empty documents, similarity score should be 0 or 1"

def test_similarity_empty_and_nonempty_docs():
    doc1 = "This is a test document"
    doc2 = ""
    result = plag.similarity(doc1, doc2)
    assert np.isclose(result, 0), "Test failed: For an empty and a non-empty document, similarity score should be 0"

def test_similarity_special_chars():
    doc1 = "@@@@!!!!"
    doc2 = "####!!!!!"
    result = plag.similarity(doc1, doc2)
    assert isinstance(result, np.ndarray), "Test failed: For documents with special characters, similarity function should not throw an error"

def test_similarity_non_english():
    doc1 = "Ce est un test document"
    doc2 = "Ce est un document échantillon"
    result = plag.similarity(doc1, doc2)
    assert isinstance(result, np.ndarray), "Test failed: For non-English documents, similarity function should not throw an error"

def test_similarity_numbers():
    doc1 = "123 456"
    doc2 = "123 789"
    result = plag.similarity(doc1, doc2)
    assert isinstance(result, np.ndarray), "Test failed: For documents with numerical characters, similarity function should not throw an error"

def test_similarity_large_texts():
    doc1 = "This is a test document" * 100000
    doc2 = "This is a sample document" * 100000
    result = plag.similarity(doc1, doc2)
    assert isinstance(result, np.ndarray), "Test failed: For large documents, similarity function should not crash due to memory issues"

def test_similarity_different_sizes():
    doc1 = "This is a test document"
    doc2 = "This is a sample document" * 10000
    result = plag.similarity(doc1, doc2)
    assert isinstance(result, np.ndarray), "Test failed: For documents with significantly different sizes, similarity function should not throw an error"
