# Workflow Diagrams

This document contains workflow diagrams that illustrate the processing pipeline in EZStitcher.

## Overall Processing Pipeline

```
┌─────────────────┐
│ process_plate_auto │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PlateProcessor  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Auto-detection  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ File Renaming   │◄───── Optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Z-Stack Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Processing│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stitching       │
└─────────────────┘
```

## Z-Stack Processing Pipeline

```
┌─────────────────┐
│ ZStackProcessor │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Z-Stack Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Focus Detection │◄───── Optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Projection      │◄───── Optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reference Generation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Position Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Per-Plane Stitching│◄───── Optional
└─────────────────┘
```

## File Renaming Workflow

```
┌─────────────────┐
│ rename_files_with_consistent_padding │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Find Image Files│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Map Original to │
│ Padded Filenames│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check for       │
│ Conflicts       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rename Files    │◄───── Skip if dry_run=True
└─────────────────┘
```

## Microscope Type Auto-Detection Workflow

```
┌─────────────────┐
│ Auto-detection  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check for Opera │
│ Phenix Index.xml│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check for       │
│ ImageXpress     │
│ File Patterns   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Initialize      │
│ Appropriate     │
│ FilenameParser  │
└─────────────────┘
```

## Stitching Workflow

```
┌─────────────────┐
│ Stitcher        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Images     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocess      │
│ Images          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │
│ Composite       │◄───── If multiple reference channels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detect Positions│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Assemble Image  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Stitched   │
│ Image           │
└─────────────────┘
```