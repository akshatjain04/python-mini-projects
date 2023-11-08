import pytest
import numpy as np
from collections import Counter

def vectorize(text):
    # Simple vectorize function assuming words are separated by spaces
    # And not performing any special character or case handling
    transformed = Counter(text.lower().strip().split())
    return np.array(list(transformed.values()))

@pytest.fixture
def vectorizer():
    return vectorize

def test_empty_text(vectorizer):
    assert np.array_equal(vectorizer(''), np.array([])), "Expected empty array for empty text."

def test_single_unique_word(vectorizer):
    assert np.array_equal(vectorizer('only'), np.array([1])), "Expected array with one element for single unique word."

def test_multiple_unique_words(vectorizer):
    result = vectorizer('word1 word2')
    assert len(result.shape) == 1, "Expected result should be an 1-d array."
    assert result.size == 2, "Expected array with size equals to number of unique words."

def test_same_word_repeated(vectorizer):
    assert np.array_equal(vectorizer('word word word'), np.array([1])), "Expected array with one element for repeated same word."

def test_real_life_example(vectorizer):
    # A real life text example
    real_life_text = "The quick brown fox jumps over the lazy dog."
    unique_words = len(set(real_life_text.split()))
    assert vectorizer(real_life_text).size == unique_words, "Expected array with size equals to number of unique words in real life examples."

def test_text_with_special_characters(vectorizer):
    result = vectorizer('word! word? "word", word.')
    assert result.size == 1, "Expected array with one element for text with special characters."

def test_text_with_digits(vectorizer):
    assert vectorizer('123 456 789').size == 3, "Expected array with size equals to number of unique digits in text."
    
def test_non_english_text(vectorizer):
    assert vectorizer('词 词 词').size == 1, "Expected array with one element for non-english text."

def test_text_with_uppercase_lowercase(vectorizer):
    result = vectorizer('Word word WoRd')
    assert result.size == 1, "Expected array with one element for text with mixed case letters."

def test_large_text(vectorizer):
    large_text = 'word ' * 10000
    assert vectorizer(large_text).size == 1, "Expected array with one element for large text."

