Changelog
=========

.. note::
   EZStitcher is currently in active development and has not yet had an official release.

   The changelog will be populated with detailed information for each official release once the project reaches a stable version.

   For the latest development changes, please refer to the commit history in the GitHub repository.

Development Changes
-----------------

Pipeline Architecture Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Function Tuple Support**: Added support for specifying functions as tuples of (function, kwargs) in the Step class. This allows for more concise and clearer code by combining functions and their arguments into a single parameter. The feature supports:

  - Single function tuples: ``func=(my_function, {'param1': value1})``
  - Lists of function tuples: ``func=[(func1, args1), (func2, args2)]``
  - Dictionaries of function tuples: ``func={"1": (func1, args1), "2": (func2, args2)}``
  - Mixed function types: ``func=[func1, (func2, args2), func3]``

- **Removed Deprecated Features**: Removed the ``processing_args`` parameter from the Step class and all related code. All examples and documentation have been updated to use function tuples instead.
