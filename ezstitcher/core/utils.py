"""
Utility functions for the EZStitcher package.
"""

import threading
import time
import functools
import logging
from collections import defaultdict
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger(__name__)

# Global thread activity tracking
thread_activity = defaultdict(list)
active_threads = set()
thread_lock = threading.Lock()

def get_thread_activity() -> Dict[int, List[Dict[str, Any]]]:
    """
    Get the current thread activity data.

    Returns:
        Dict mapping thread IDs to lists of activity records
    """
    return thread_activity

def get_active_threads() -> set:
    """
    Get the set of currently active thread IDs.

    Returns:
        Set of active thread IDs
    """
    return active_threads

def clear_thread_activity():
    """Clear all thread activity data."""
    with thread_lock:
        thread_activity.clear()
        active_threads.clear()

def track_thread_activity(func: Optional[Callable] = None, *, log_level: str = "info"):
    """
    Decorator to track thread activity for a function.

    Args:
        func: The function to decorate
        log_level: Logging level to use ("debug", "info", "warning", "error")

    Returns:
        Decorated function that tracks thread activity
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get thread information
            thread_id = threading.get_ident()
            thread_name = threading.current_thread().name

            # Record thread start time
            start_time = time.time()

            # Extract function name and arguments for context
            func_name = f.__name__
            # Get the first argument if it's a method (self or cls)
            context = ""
            if args and hasattr(args[0], "__class__"):
                if hasattr(args[0].__class__, func_name):
                    # It's likely a method, extract class name
                    context = f"{args[0].__class__.__name__}."

            # Extract well information if present in kwargs or args
            well = kwargs.get('well', None)
            if well is None and len(args) > 1 and isinstance(args[1], str):
                # Assume second argument might be well in methods like process_well(self, well, ...)
                well = args[1]

            # Add this thread to active threads
            with thread_lock:
                active_threads.add(thread_id)
                # Record the number of active threads at this moment
                thread_activity[thread_id].append({
                    'well': well,
                    'thread_name': thread_name,
                    'time': time.time(),
                    'action': 'start',
                    'function': f"{context}{func_name}",
                    'active_threads': len(active_threads)
                })

            # Log the start of the function
            log_func = getattr(logger, log_level.lower())
            log_func(f"Thread {thread_name} (ID: {thread_id}) started {context}{func_name} for well {well}")
            log_func(f"Active threads: {len(active_threads)}")

            try:
                # Call the original function
                result = f(*args, **kwargs)
                return result
            finally:
                # Record thread end time
                end_time = time.time()
                duration = end_time - start_time

                # Remove this thread from active threads
                with thread_lock:
                    active_threads.remove(thread_id)
                    # Record the number of active threads at this moment
                    thread_activity[thread_id].append({
                        'well': well,
                        'thread_name': thread_name,
                        'time': time.time(),
                        'action': 'end',
                        'function': f"{context}{func_name}",
                        'duration': duration,
                        'active_threads': len(active_threads)
                    })

                log_func(f"Thread {thread_name} (ID: {thread_id}) finished {context}{func_name} for well {well} in {duration:.2f} seconds")
                log_func(f"Active threads: {len(active_threads)}")

        return wrapper

    # Handle both @track_thread_activity and @track_thread_activity(log_level="debug")
    if func is None:
        return decorator
    return decorator(func)

def analyze_thread_activity():
    """
    Analyze thread activity data and return a report.

    Returns:
        Dict containing analysis results
    """
    max_concurrent = 0
    thread_starts = []
    thread_ends = []

    for thread_id, activities in thread_activity.items():
        for activity in activities:
            max_concurrent = max(max_concurrent, activity['active_threads'])
            if activity['action'] == 'start':
                thread_starts.append((
                    activity.get('well'),
                    activity['thread_name'],
                    activity['time'],
                    activity.get('function', '')
                ))
            else:  # 'end'
                thread_ends.append((
                    activity.get('well'),
                    activity['thread_name'],
                    activity['time'],
                    activity.get('duration', 0),
                    activity.get('function', '')
                ))

    # Sort by time
    thread_starts.sort(key=lambda x: x[2])
    thread_ends.sort(key=lambda x: x[2])

    # Find overlapping time periods
    overlaps = []
    for i, (well1, thread1, start1, func1) in enumerate(thread_starts):
        # Find the end time for this thread
        end1 = None
        for w, t, end, d, f in thread_ends:
            if t == thread1 and w == well1 and f == func1:
                end1 = end
                break

        if end1 is None:
            continue  # Skip if we can't find the end time

        # Check for overlaps with other threads
        for j, (well2, thread2, start2, func2) in enumerate(thread_starts):
            if i == j or thread1 == thread2:  # Skip same thread
                continue

            # Find the end time for the other thread
            end2 = None
            for w, t, end, d, f in thread_ends:
                if t == thread2 and w == well2 and f == func2:
                    end2 = end
                    break

            if end2 is None:
                continue  # Skip if we can't find the end time

            # Check if there's an overlap
            if start1 < end2 and start2 < end1:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:
                    overlaps.append({
                        'thread1': thread1,
                        'well1': well1,
                        'function1': func1,
                        'thread2': thread2,
                        'well2': well2,
                        'function2': func2,
                        'duration': overlap_duration
                    })

    return {
        'max_concurrent': max_concurrent,
        'thread_starts': thread_starts,
        'thread_ends': thread_ends,
        'overlaps': overlaps
    }

def print_thread_activity_report():
    """Print a detailed report of thread activity."""
    analysis = analyze_thread_activity()

    print("\n" + "=" * 80)
    print("Thread Activity Report")
    print("=" * 80)

    print("\nThread Start Events:")
    for well, thread_name, time_val, func in analysis['thread_starts']:
        print(f"Thread {thread_name} started {func} for well {well} at {time_val:.2f}")

    print("\nThread End Events:")
    for well, thread_name, time_val, duration, func in analysis['thread_ends']:
        print(f"Thread {thread_name} finished {func} for well {well} at {time_val:.2f} (duration: {duration:.2f}s)")

    print("\nOverlap Analysis:")
    for overlap in analysis['overlaps']:
        print(f"Threads {overlap['thread1']} and {overlap['thread2']} overlapped for {overlap['duration']:.2f}s")
        print(f"  {overlap['thread1']} was processing {overlap['function1']} for well {overlap['well1']}")
        print(f"  {overlap['thread2']} was processing {overlap['function2']} for well {overlap['well2']}")

    print(f"\nFound {len(analysis['overlaps'])} thread overlaps")
    print(f"Maximum concurrent threads: {analysis['max_concurrent']}")
    print("=" * 80)

    return analysis


import numpy as np


def prepare_patterns_and_functions(patterns, processing_funcs, component='default'):
    """
    Prepare patterns, processing functions, and processing args for processing.

    This function handles three main tasks:
    1. Ensuring patterns are in a component-keyed dictionary format
    2. Determining which processing functions to use for each component
    3. Determining which processing args to use for each component

    Args:
        patterns (list or dict): Patterns to process, either as a flat list or grouped by component
        processing_funcs (callable, list, dict, tuple, optional): Processing functions to apply.
            Can be a single callable, a tuple of (callable, kwargs), a list of either,
            or a dictionary mapping component values to any of these.
        component (str): Component name for grouping (only used for clarity in the result)

    Returns:
        tuple: (grouped_patterns, component_to_funcs, component_to_args)
            - grouped_patterns: Dictionary mapping component values to patterns
            - component_to_funcs: Dictionary mapping component values to processing functions
            - component_to_args: Dictionary mapping component values to processing args
    """
    # Ensure patterns are in a dictionary format
    # If already a dict, use as is; otherwise wrap the list in a dictionary
    grouped_patterns = patterns if isinstance(patterns, dict) else {component: patterns}

    # Initialize dictionaries for functions and args
    component_to_funcs = {}
    component_to_args = {}

    # Helper function to extract function and args from a function item
    def extract_func_and_args(func_item):
        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            # It's a (function, kwargs) tuple
            return func_item[0], func_item[1]
        if callable(func_item):
            # It's just a function, use default args
            return func_item, {}
        # Invalid function item
        logger.warning(
            "Invalid function item: %s. Expected callable or (callable, kwargs) tuple.",
            str(func_item)
        )
        # Return a dummy function that returns the input unchanged
        return lambda x, **kwargs: x, {}

    for comp_value in grouped_patterns.keys():
        # Get functions and args for this component
        if isinstance(processing_funcs, dict) and comp_value in processing_funcs:
            # Direct mapping for this component
            func_item = processing_funcs[comp_value]
        elif isinstance(processing_funcs, dict) and component == 'channel':
            # For channel grouping, use the channel-specific function if available
            func_item = processing_funcs.get(comp_value, processing_funcs)
        else:
            # Use the same function for all components
            func_item = processing_funcs

        # Extract function and args
        if isinstance(func_item, list):
            # List of functions or function tuples
            component_to_funcs[comp_value] = func_item
            # For lists, we'll extract args during processing
            component_to_args[comp_value] = {}
        else:
            # Single function or function tuple
            func, args = extract_func_and_args(func_item)
            component_to_funcs[comp_value] = func
            component_to_args[comp_value] = args

    return grouped_patterns, component_to_funcs, component_to_args


def stack(single_image_func: Callable) -> Callable[[List[np.ndarray], Optional[Dict[str, Any]]], List[np.ndarray]]:
    """Wraps a function designed for single images to operate on a stack (list) of images.

    Args:
        single_image_func: A function that processes a single numpy array image
                           and returns a processed numpy array image.

    Returns:
        A new function that accepts a list of images and keyword arguments,
        applies the original function to each image in the list, and returns
        a list of the processed images.
    """
    @functools.wraps(single_image_func)
    def stack_wrapper(images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """Applies the wrapped single-image function to each image in the stack."""
        processed_stack = []
        if not images:
            return processed_stack # Return empty list if input is empty

        for img in images:
            try:
                # Pass only the image and any relevant kwargs accepted by the original function
                # Inspecting signature might be needed for robustness, but start simple
                processed_img = single_image_func(img, **kwargs)
                if processed_img is not None:
                    processed_stack.append(processed_img)
                else:
                     logger.warning(f"Function {single_image_func.__name__} returned None for an image. Skipping.")
            except Exception as e:
                logger.error(f"Error applying {single_image_func.__name__} to an image in the stack: {e}. Skipping image.")
        return processed_stack

    # Attempt to give the wrapper a more informative name
    try:
        stack_wrapper.__name__ = f"stacked_{single_image_func.__name__}"
    except AttributeError:
        pass # Some callables might not have __name__

    return stack_wrapper

