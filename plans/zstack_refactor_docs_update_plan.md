# Z-Stack Refactor Documentation Update Plan

This plan outlines the steps to update the documentation and README to reflect the new modular, object-oriented API after refactoring the monolithic `ZStackProcessor` class.

---

## 1. API Reference (`docs/source/api/`)

- **Deprecate or remove** `zstack_processor.rst` which documents the old monolithic class.
- **Create new API documentation files** for each new component:
  - `zstack_detector.rst`
  - `zstack_organizer.rst`
  - `zstack_metadata.rst`
  - `zstack_focus_selector.rst`
  - `zstack_projector.rst`
  - `reference_projection_generator.rst`
  - `position_file_manager.rst`
  - `zplane_stitcher.rst`
  - `file_resolver.rst`
  - `zstack_pipeline.rst` (or equivalent orchestrator)
- **Update `api/index.rst`** to include these new modules.
- Ensure each `.rst` file uses `.. automodule::` with `:members:` and appropriate options.

---

## 2. README.md

- **Revise the "Class-Based Architecture" section:**
  - Replace the single `ZStackProcessor` with a list of new modular components.
  - Add a brief description of each component's responsibility.
- **Add an architecture diagram** (e.g., Mermaid) illustrating the new modular design.
- **Clarify** that typical users interact via `PlateProcessor` and `process_plate_folder()`.
- **Mention** that advanced users can use individual components directly for custom workflows.

---

## 3. Examples and User Guide

- **Keep existing examples mostly unchanged** since they rely on high-level APIs (`process_plate_folder`, `PlateProcessor`).
- **Optionally add advanced examples** demonstrating direct use of new components.
- **Update any prose references** to `ZStackProcessor` to reflect the modular design.
- Review `docs/source/examples/zstack_processing.rst` and `docs/source/user_guide.rst` for necessary updates.

---

## 4. Summary

This plan ensures the documentation accurately reflects the new, cleaner, modular API, while maintaining backward compatibility for most users. It will improve clarity for both new and advanced users, and facilitate future maintenance.