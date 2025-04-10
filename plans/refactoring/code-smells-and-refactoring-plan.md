# Code Smells and Refactoring Suggestions

---

## 1. Raw File and Directory Operations

### Smell:
- Direct use of `Path` methods like `.mkdir()`, `.glob()`, `.exists()`, `.is_dir()`
- Scattered across `PlateProcessor`, `ZStackProcessor`, and other modules

### Suggestion:
- Replace with **`FileSystemManager`** instance methods:
  - `ensure_directory()`
  - `list_image_files()`
  - `find_htd_file()`
  - `parse_htd_file()`
- **Inject `FileSystemManager`** into all classes that need file I/O
- **Benefits:** Centralizes file handling, improves error handling, easier to mock/test

---

## 2. Manual Filename Parsing

### Smell:
- Regex and string slicing scattered in multiple places
- Hardcoded assumptions about filename formats

### Suggestion:
- Use the **`FilenameParser`** interface and its subclasses:
  - `ImageXpressFilenameParser`
  - `OperaPhenixFilenameParser`
- Select parser via config or auto-detection
- Replace all manual parsing with calls to:
  - `parse_well()`
  - `parse_site()`
  - `parse_z_index()`
  - `parse_channel()`
- **Benefits:** Consistent, extensible parsing; easier to add new formats

---

## 3. Ad-hoc Pattern Matching and Grouping

### Smell:
- Grouping images by filename suffixes or glob patterns
- Logic duplicated across modules

### Suggestion:
- Use the **`PatternMatcher`** class:
  - `group_images()` method to group by well, site, z, channel
- Refactor pipeline to **operate on these groups**
- **Benefits:** Cleaner grouping, easier to extend, less duplication

---

## 4. Preprocessing Function Handling

### Smell:
- Assumes functions are either single-image or stack-aware
- No uniform interface

### Suggestion:
- Use **`ZStackProcessor._adapt_function()`** to wrap all preprocessing functions
- Always pass **stacks** to adapted functions
- **Benefits:** Uniform interface, supports both function types seamlessly

---

## 5. Microscope Detection and Conversion

### Smell:
- Manual detection logic and renaming scattered
- Risk of inconsistent handling

### Suggestion:
- Use **`detect_parser()`** to auto-detect microscope type
- Use `OperaPhenixFilenameParser.rename_all_files_in_directory()` for conversion
- Store parser in config or context
- **Benefits:** Cleaner, more reliable format handling

---

## 6. Configuration Fragmentation

### Smell:
- Many parameters passed explicitly
- Risk of inconsistency and long signatures

### Suggestion:
- Use **config objects** (`PlateProcessorConfig`, `StitcherConfig`, etc.) everywhere
- Pass config objects instead of individual parameters
- **Benefits:** Centralized, validated configuration

---

## 7. Monolithic Pipeline Methods

### Smell:
- Large methods like `PlateProcessor.run()`
- Hard to test, maintain, or extend

### Suggestion:
- Break into smaller, well-named methods:
  - `_detect_and_rename()`
  - `_group_tiles()`
  - `_preprocess_groups()`
  - `_apply_z_reference()`
  - `_generate_stitching_positions()`
  - `_final_stitching()`
- **Benefits:** Easier testing, clearer logic, better extensibility

---

## Summary

Refactoring these areas will:

- Reduce code duplication
- Improve maintainability
- Enhance extensibility
- Eliminate code smells
- Enable a cleaner, more modular, and future-proof codebase

---