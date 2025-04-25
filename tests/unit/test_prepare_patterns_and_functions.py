"""
Test the prepare_patterns_and_functions function.
"""

import pytest
from ezstitcher.core.utils import prepare_patterns_and_functions


# Define test functions
def func1(images):
    """Function 1."""
    return images

def func2(images):
    """Function 2."""
    return images

def func3(images):
    """Function 3."""
    return images


class TestPreparePatternsAndFunctions:
    """Test the prepare_patterns_and_functions function."""

    def test_single_function_single_args(self):
        """Test with a single function and single args dictionary."""
        patterns = ["pattern1", "pattern2"]
        func = (func1, {"param": "value"})

        grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
            patterns, func
        )

        assert "default" in grouped_patterns
        assert grouped_patterns["default"] == patterns
        assert component_to_funcs["default"] == func1
        assert component_to_args["default"] == {"param": "value"}

    def test_list_functions_list_args(self):
        """Test with a list of functions and a list of args dictionaries."""
        patterns = ["pattern1", "pattern2"]
        funcs = [
            (func1, {"param1": "value1"}),
            (func2, {"param2": "value2"}),
            (func3, {"param3": "value3"})
        ]

        grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
            patterns, funcs
        )

        assert "default" in grouped_patterns
        assert grouped_patterns["default"] == patterns
        assert component_to_funcs["default"] == funcs
        # component_to_args will contain empty dicts for each function in the list
        assert isinstance(component_to_args["default"], dict)

    def test_dict_functions_dict_args(self):
        """Test with a dictionary of functions and a dictionary of args."""
        patterns = {
            "1": ["pattern1_1", "pattern1_2"],
            "2": ["pattern2_1", "pattern2_2"]
        }
        funcs = {
            "1": (func1, {"param1": "value1"}),
            "2": (func2, {"param2": "value2"})
        }

        grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
            patterns, funcs
        )

        assert "1" in grouped_patterns and "2" in grouped_patterns
        assert grouped_patterns["1"] == patterns["1"]
        assert grouped_patterns["2"] == patterns["2"]
        assert component_to_funcs["1"] == func1
        assert component_to_funcs["2"] == func2
        assert component_to_args["1"] == {"param1": "value1"}
        assert component_to_args["2"] == {"param2": "value2"}

    def test_dict_list_functions_dict_list_args(self):
        """Test with a dictionary of lists of functions and a dictionary of lists of args."""
        patterns = {
            "1": ["pattern1_1", "pattern1_2"],
            "2": ["pattern2_1", "pattern2_2"]
        }
        funcs = {
            "1": [
                (func1, {"param1a": "value1a"}),
                (func2, {"param1b": "value1b"})
            ],
            "2": [
                (func3, {"param2": "value2"})
            ]
        }

        grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
            patterns, funcs
        )

        assert "1" in grouped_patterns and "2" in grouped_patterns
        assert grouped_patterns["1"] == patterns["1"]
        assert grouped_patterns["2"] == patterns["2"]
        # For lists of functions, component_to_funcs will contain the list
        assert component_to_funcs["1"] == funcs["1"]
        assert component_to_funcs["2"] == funcs["2"]
        # component_to_args will contain empty dicts for lists of functions
        assert isinstance(component_to_args["1"], dict)
        assert isinstance(component_to_args["2"], dict)

    def test_dict_functions_no_args(self):
        """Test with a dictionary of functions and no args."""
        patterns = {
            "1": ["pattern1_1", "pattern1_2"],
            "2": ["pattern2_1", "pattern2_2"]
        }
        funcs = {
            "1": func1,  # Plain function without args
            "2": func2   # Plain function without args
        }

        grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
            patterns, funcs
        )

        assert "1" in grouped_patterns and "2" in grouped_patterns
        assert grouped_patterns["1"] == patterns["1"]
        assert grouped_patterns["2"] == patterns["2"]
        assert component_to_funcs["1"] == func1
        assert component_to_funcs["2"] == func2
        # With plain functions, component_to_args will contain empty dicts
        assert component_to_args["1"] == {}
        assert component_to_args["2"] == {}

    def test_dict_functions_with_same_args(self):
        """Test with a dictionary of function tuples that have the same args."""
        patterns = {
            "1": ["pattern1_1", "pattern1_2"],
            "2": ["pattern2_1", "pattern2_2"]
        }
        funcs = {
            "1": (func1, {"param": "value"}),
            "2": (func2, {"param": "value"})
        }

        grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
            patterns, funcs
        )

        assert "1" in grouped_patterns and "2" in grouped_patterns
        assert grouped_patterns["1"] == patterns["1"]
        assert grouped_patterns["2"] == patterns["2"]
        assert component_to_funcs["1"] == func1
        assert component_to_funcs["2"] == func2
        assert component_to_args["1"] == {"param": "value"}
        assert component_to_args["2"] == {"param": "value"}
