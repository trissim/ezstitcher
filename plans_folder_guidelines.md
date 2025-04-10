# Plans Folder Guidelines

## Key Points

1. **Dedicated Location**: All thought files must be stored in the `plans/` folder
2. **Consistent References**: Always include the `plans/` prefix when referencing files
3. **Hierarchical Organization**: Use subfolders within `plans/` for complex domains

## Benefits of Using the Plans Folder

- **Centralized Knowledge**: All planning documents are in one location
- **Reduced Clutter**: Keeps the main project directories clean
- **Easier Navigation**: Simplifies finding and referencing thought files
- **Consistent Structure**: Establishes a standard location for all planning activities

## Implementation Details

### Creating the Plans Folder

- The `plans/` folder should be created at the root of the project
- It should be included in version control
- It should have a README.md explaining its purpose and structure

### Organizing Subfolders

For complex projects, organize the `plans/` folder with domain-specific subfolders:

```
plans/
├── README.md
├── architecture/
│   ├── system-design-plan.md
│   └── component-interaction-plan.md
├── features/
│   ├── user-authentication-plan.md
│   └── payment-processing-plan.md
└── refactoring/
    ├── code-cleanup-plan.md
    └── performance-optimization-plan.md
```

### File Naming Within Plans Folder

- All files should follow the established naming conventions
- Always include the `plans/` prefix when referencing files in discussions
- Use the `_complete` suffix for completed thought files

## Common Mistakes to Avoid

1. Creating thought files outside the `plans/` folder
2. Referencing thought files without the `plans/` prefix
3. Mixing implementation code with planning documents
4. Neglecting to update the status of thought files

## Example Usage

When discussing a plan:
"I've outlined the authentication system in `plans/auth-system-plan.md` and broken down the password reset flow in `plans/auth/password-reset-plan.md`."

When referencing in another thought file:
"As discussed in [[plans/auth-system-plan.md#security-considerations]], we need to implement rate limiting."
