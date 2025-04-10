# Pydantic Validator Upgrade Plan

```
Status: Complete
Progress: 100%
Last Updated: 2023-04-09
Dependencies: []
```

## Problem Analysis

Based on the codebase review, one of the most pressing issues is the use of deprecated Pydantic V1 style validators in the configuration system. The codebase currently uses `@validator` decorators, which are deprecated in Pydantic V2 and will be removed in V3.0.

### Current Implementation

The current implementation in `pydantic_config.py` uses the deprecated V1 style validators:

```python
@validator('margin_ratio')
def validate_margin_ratio(cls, v):
    """Validate that margin_ratio is between 0 and 1."""
    if not 0 <= v <= 1:
        raise ValueError(f"margin_ratio must be between 0 and 1, got {v}")
    return v
```

### Requirements

1. Update all validators to use Pydantic V2 style `@field_validator` decorators
2. Maintain the same validation logic
3. Ensure all tests pass after the changes
4. Update any related documentation

## High-Level Solution

The solution involves:

1. Identifying all instances of `@validator` decorators in the codebase
2. Replacing them with the new `@field_validator` decorators
3. Updating the validator method signatures as needed
4. Running tests to ensure the changes don't break existing functionality

### Pydantic V2 Migration Guide

According to the [Pydantic V2 Migration Guide](https://docs.pydantic.dev/latest/migration/), the changes needed are:

1. Import `field_validator` instead of `validator`:
   ```python
   from pydantic import field_validator  # instead of validator
   ```

2. Update the decorator and method signature:
   ```python
   @field_validator('field_name')
   @classmethod  # field_validator requires the classmethod decorator
   def validate_field(cls, v, info):
       # validation logic
       return v
   ```

3. The `info` parameter is optional but provides access to additional validation context

## Implementation Details

### Files to Modify

Based on the codebase review, the main file to modify is:
- `ezstitcher/core/pydantic_config.py`

### Validators to Update

Based on the warnings, we need to update the following validators:

1. `StitcherConfig.validate_margin_ratio` (line 34)
2. `StitcherConfig.validate_overlap` (line 41)
3. `FocusAnalyzerConfig.validate_method` (line 62)
4. `ZStackProcessorConfig.validate_focus_method` (line 159)
5. `ZStackProcessorConfig.validate_additional_projections` (line 167)
6. `ZStackProcessorConfig.validate_z_reference_function` (line 176)
7. `ZStackProcessorConfig.validate_projection_types` (line 185)
8. `ZStackProcessorConfig.validate_reference_method` (line 195)
9. `ZStackProcessorConfig.validate_stitch_z_reference` (line 204)
10. `PlateProcessorConfig.validate_reference_channels` (line 347)

We also need to update the class-based `Config` classes to use `model_config` with `ConfigDict`.

### Sample Implementation

Here's a sample of how the updated code should look:

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict

class StitcherConfig(BaseModel):
    # Fields remain the same
    tile_overlap: float = Field(6.5, description="Default percentage overlap between tiles")
    # ...

    @field_validator('margin_ratio')
    @classmethod
    def validate_margin_ratio(cls, v):
        """Validate that margin_ratio is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"margin_ratio must be between 0 and 1, got {v}")
        return v

    # For multiple fields
    @field_validator('tile_overlap', 'tile_overlap_x', 'tile_overlap_y')
    @classmethod
    def validate_overlap(cls, v):
        """Validate that overlap percentages are reasonable."""
        if v is not None and not 0 <= v <= 50:
            raise ValueError(f"Tile overlap should be between 0 and 50 percent, got {v}")
        return v

# Replace class Config with model_config
class ImagePreprocessorConfig(BaseModel):
    preprocessing_funcs: Dict[str, Callable] = Field(default_factory=dict,
                                                  description="Preprocessing functions by channel")
    composite_weights: Optional[Dict[str, float]] = Field(None,
                                                       description="Weights for creating composite images")

    # Replace this:
    # class Config:
    #     arbitrary_types_allowed = True

    # With this:
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

### Testing Strategy

1. Run the existing tests to ensure they pass with the current implementation
2. Make the changes to one validator at a time
3. Run the tests after each change to identify any issues
4. Fix any issues that arise
5. Continue until all validators are updated

## Validation

The changes should not affect the behavior of the configuration system, as we're only updating the syntax of the validators, not their logic. The tests should continue to pass after the changes.

## Next Steps

1. Examine the current implementation in detail
2. Create a backup of the current code
3. Implement the changes one validator at a time
4. Run tests after each change
5. Document any issues encountered and their solutions

## Completion Summary

**Date**: 2023-04-09

### Changes Made

1. Updated all `@validator` decorators to `@field_validator` and added `@classmethod` decorators
2. Replaced all `class Config` classes with `model_config = ConfigDict`
3. Fixed all Pydantic deprecation warnings

### Results

- All tests are passing
- No more Pydantic deprecation warnings
- The codebase is now compatible with Pydantic V2

### Lessons Learned

- The migration from Pydantic V1 to V2 is straightforward but requires careful attention to detail
- The `@classmethod` decorator is required for all `@field_validator` methods
- The `model_config` attribute with `ConfigDict` is a cleaner way to configure Pydantic models
