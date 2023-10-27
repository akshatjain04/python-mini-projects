# Test generated by RoostGPT for test MiniProjects using AI Type Azure Open AI and AI Model roost-gpt4-32k

# Import necessary libraries
from make_art import print_out_ascii
import numpy as np
import sys
import io

# TODO: Update symbols_list with actual symbols you are using

symbols_list = ['*', '#']

def test_Print_out_ascii_d593277e76():
    try:
        captured_output = io.StringIO()
        sys.stdout = captured_output
        array1 = np.array([[1, 2], [3, 4]])
        print_out_ascii(array1)
        output1 = captured_output.getvalue()
        assert output1 == '#*\n##\n', "Test Case 1 Failed"

        captured_output = io.StringIO()
        sys.stdout = captured_output
        array2 = np.array([[1, 0, 1], [2, 0, 2], [1, 1, 1]])
        print_out_ascii(array2)
        output2 = captured_output.getvalue()
        assert output2 == '#*#\n**\n###\n', "Test Case 2 Failed"
        
        print("All test cases pass")

    except Exception as e:
        print("Exception occurred during testing: ", e)
        assert False, "Test Case Failed"

# Call the function to test
test_Print_out_ascii_d593277e76()
