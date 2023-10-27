# Test generated by RoostGPT for test MiniProjects using AI Type Azure Open AI and AI Model roost-gpt4-32k

import pytest
from biling_system import BillSystem  # assuming biling_system is the file where this class is defined

@pytest.fixture
def setup_bill_system():
    # Creating a mock of the BillSystem class
    bill_system = BillSystem()
    
    bill_system.c_name = ""
    bill_system.c_phone = ""
    bill_system.medical_price = "Rs. 0.0"
    bill_system.grocery_price = "Rs. 0.0"
    bill_system.cold_drinks_price = "Rs. 0.0"

    # TODO - Initialize all necessary attributes here with your own test data

    return bill_system


def test_Bill_area_2d56362e02(setup_bill_system):
    
    # Bill_system without customer details
    with pytest.raises(Exception) as e_info:
        setup_bill_system.bill_area()
    assert "Customer Details Are Must" in str(e_info.value), "Test for missing customer details failed"
    
    # Bill_system without any product purchase
    setup_bill_system.c_name = "Test_name"
    setup_bill_system.c_phone = "0123456789"
    
    with pytest.raises(Exception) as e_info:
        setup_bill_system.bill_area()
    assert "No Product Purchased" in str(e_info.value), "Test for no product purchased failed"

    # TODO - Add more test cases for other conditions

