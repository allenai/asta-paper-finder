# 🧰 ai2i.common

A library of common utilities and algorithms for Python applications.

## Overview

`ai2i.common` provides a collection of reusable components and utilities that are commonly needed across different
applications, with features like:

- Asynchronous programming utilities
- Batch processing tools
- Time-related utilities
- Data structure utilities
- Multi-Armed Bandit (MAB) algorithms

## Basic Usage

### Asynchronous Utilities

```python
from ai2i.common.utils.asyncio import custom_gather

# Run tasks in parallel (default behavior)
results = await custom_gather(task1, task2, task3, force_deterministic=False)

# Run tasks serially for deterministic results
results = await custom_gather(task1, task2, task3, force_deterministic=True)
```

### Batch Processing

```python
from ai2i.common.utils.batch import batch_process, with_batch

# Process items in batches
results = await batch_process(
    items=large_list_of_items,
    batch_size=100,
    process_func=async_process_function,
    max_concurrency=10
)


# Use as a decorator
@with_batch(batch_size=100, max_concurrency=10)
async def process_items(items):
    # Process a batch of items
    return [await process_item(item) for item in items]


# Call the decorated function with a large list
results = await process_items(large_list_of_items)
```

### Time Utilities

```python
from ai2i.common.utils.time import get_utc_time, timing, atiming

# Get current UTC time
now = get_utc_time()


# Time a synchronous function
@timing
def expensive_operation():
    # Function execution time will be logged
    pass


# Time an asynchronous function
@atiming
async def async_expensive_operation():
    # Function execution time will be logged
    pass
```

### Data Structure Utilities

```python
from ai2i.common.utils.data_struct import SortedSet

# Create a set that iterates in sorted order
sorted_set = SortedSet([3, 1, 4, 1, 5, 9, 2, 6])

# Iteration will be in sorted order
for item in sorted_set:
    print(item)  # Will print: 1, 2, 3, 4, 5, 6, 9
```

### Value Utilities

```python
from ai2i.common.utils.value import ValueNotSet


# Use ValueNotSet when None is a valid value
def process_data(data, optional_param=ValueNotSet.instance()):
    if optional_param is ValueNotSet.instance():
        # Use default behavior
        pass
    else:
        # Use provided value (which could be None)
        pass
```

## Advanced Features

## Development

### Setup Development Environment

```shell script
make sync-dev
```

### Testing

```shell script
make test
# -or-
make test-cov  # With coverage
```

### Code Quality

```shell script
make style  # Run format check, lint and type-check
make fix    # Automatically fix style issues
```
