import pytest
import random_password_gen as rpg
    
def test_generate_pass_min_length():
    password = rpg.generate_pass(1, 'abc')
    assert len(password) == 1
    
def test_generate_pass_max_length():
    password = rpg.generate_pass(1000, 'abc')
    assert len(password) == 1000

def test_generate_pass_custom_array():
    char_array = 'abc'
    password = rpg.generate_pass(5, char_array)
    assert all(char in char_array for char in password)

def test_generate_pass_alpha_true():
    password = rpg.generate_pass(10, 'abc', True)
    assert any(char.isupper() for char in password) and any(char.islower() for char in password)
    
def test_generate_pass_alpha_false():
    password = rpg.generate_pass(10, 'abc')
    assert all(char.islower() for char in password)
    
def test_generate_pass_unique_pass():
    pass1 = rpg.generate_pass(10, 'abc')
    pass2 = rpg.generate_pass(10, 'abc')
    assert pass1 != pass2
    
def test_generate_pass_validate_length():
    password = rpg.generate_pass(12, 'abc')
    assert len(password) == 12

def test_generate_pass_empty_array():
    with pytest.raises(ValueError):
        rpg.generate_pass(10, '')

def test_generate_pass_special_character_alpha_true():
    input_array = '@#1'
    password = rpg.generate_pass(10, 'abc@#1', True)
    assert all(char.isalpha() for char in password)
