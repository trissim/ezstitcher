# Documentation Restructuring Plan

## Current Issues
- Redundant content across multiple files
- Inconsistent information between similar sections
- Outdated references to pip installation
- Overlapping concepts in different sections
- Duplicate code examples

## Proposed Structure

### Getting Started
- **Installation** (single source of truth)
- **Quick Start** (simplified examples)
- **Core Concepts** (merged from basic_concepts and core_concepts)

### User Guide
- **Configuration** (consolidated configuration information)
- **Microscope Support** (unified microscope reference)
- **Image Processing** (comprehensive guide to image processing features)
- **Z-Stack Handling** (consolidated Z-stack information)
- **Stitching** (focused solely on stitching functionality)

### API Reference
- Keep structure but ensure no duplication with User Guide
- Focus on technical specifications rather than usage examples

### Examples
- Practical, complete examples without repeating conceptual information
- Each example should demonstrate a specific workflow

### Development
- **Contributing** (merged with testing information)
- **Architecture** (technical overview for developers)
- **Extending** (guidelines for adding new features)

### Troubleshooting
- Hierarchical organization of issues
- Clear separation between installation and runtime issues

### Appendices
- **Glossary** (technical terms)
- **File Formats** (technical specifications only)
- **Changelog** (updated with actual dates)

## Implementation Steps

1. Consolidate installation instructions
2. Merge basic_concepts and core_concepts
3. Unify file organization documentation
4. Consolidate Z-stack processing information
5. Create a unified microscope reference
6. Streamline configuration documentation
7. Reorganize examples to avoid duplication
8. Update cross-references throughout
9. Remove outdated information
10. Ensure consistency in terminology and style
