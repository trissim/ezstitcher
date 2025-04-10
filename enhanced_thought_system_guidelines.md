# Enhanced Thought File System Guidelines

## File Organization

**All thought files must be stored in a dedicated `plans/` folder**

- Store all thought files in the `plans/` folder to maintain organization
- Use subfolders within `plans/` for different domains or projects when necessary
- Never store thought files in the root directory or other project folders
- Reference the `plans/` path when linking between thought files

## Core Principles

1. **Systematic Planning**: Every problem requires a structured plan before implementation
2. **Progressive Refinement**: Plans evolve through iterations and can be revisited
3. **Hierarchical Organization**: Complex plans are broken down into manageable components
4. **Traceability**: All thoughts should be documented and easily referenced
5. **Completion Tracking**: Clear indication of progress and completion status

## Working with Plan Files

### Initial Plan Creation

For each new problem, create a comprehensive plan file that includes:

1. **Problem Analysis** (Required)
   - Clear description of the problem statement
   - Relevant context from the codebase (use context engine)
   - Constraints and requirements
   - Potential edge cases to consider

2. **High-Level Solution** (Required)
   - Architectural overview
   - Component interactions
   - Pseudo-code for key algorithms
   - Data flow diagrams when applicable

3. **Implementation Details** (Required)
   - Sample code for critical components
   - API specifications
   - Data structures and their relationships
   - Error handling strategies

4. **Validation** (Required)
   - Similarity check against existing codebase (use context engine)
   - Potential conflicts with existing functionality
   - Performance considerations
   - Testing approach

5. **References** (Optional)
   - Links to related thought files
   - External resources or documentation
   - Relevant code snippets from the existing codebase

### Naming Conventions

- Use descriptive, kebab-case names for thought files: `plans/feature-implementation-plan.md`
- Include the problem domain in the filename: `plans/authentication-refactoring-plan.md`
- For sub-plans, use a prefix indicating the parent plan: `plans/auth-refactoring-password-reset.md`
- For domain-specific subfolders: `plans/authentication/password-reset-plan.md`

## Thought Completion Tracking

### Status Indicators

- **In Progress**: Default state for any new thought file
- **Complete**: Add `_complete` suffix when all points are thoroughly addressed
- **Blocked**: Add `_blocked` suffix when progress is halted by external factors
- **Deprecated**: Add `_deprecated` suffix for thoughts no longer relevant

### Completion Process

1. **Marking Completion**:
   - Review all sections of the thought file to ensure completeness
   - Verify that all proposed solutions have been implemented or addressed
   - Add the `_complete` suffix to the filename (e.g., `plans/refactoring-plan_complete.md`)
   - Include a completion summary at the end of the file with date and outcome

2. **Revisiting Complete Thoughts**:
   - When adding minor clarifications or enhancements:
     - Maintain the `_complete` suffix
     - Add a "Revision" section with date and changes

   - When adding significant new content or tasks:
     - Remove the `_complete` suffix
     - Add a "Revision" section with date and changes
     - Add a "Pending Items" section listing new tasks
     - Re-add the `_complete` suffix only when all new items are addressed

3. **Progress Tracking**:
   - At the beginning of each thought file, include a progress tracker:
     ```
     Status: [In Progress|Complete|Blocked|Deprecated]
     Progress: [0-100%]
     Last Updated: [YYYY-MM-DD]
     Dependencies: [List of related thought files]
     ```
   - Update this tracker with each revision

## Managing Complex Plans

### Breaking Up Large Files

When a plan becomes too large (>500 lines or too complex for a single file):

1. **Create an Index File**:
   - Name it `plans/[domain]-plan-index.md`
   - Include an overview of the problem and solution
   - List all component files with brief descriptions (always with `plans/` prefix)
   - Add a status dashboard showing completion status of all components

2. **Component Files Structure**:
   - Each component file should focus on a single aspect or subsystem
   - Recommended size: 200-300 lines maximum
   - Follow a consistent structure across all component files
   - Include cross-references to related component files

3. **Naming Pattern for Component Files**:
   - Use `plans/[domain]-[component]-plan.md` format
   - Example: For an authentication system:
     - `plans/auth-plan-index.md`
     - `plans/auth-login-flow-plan.md`
     - `plans/auth-password-reset-plan.md`
     - `plans/auth-session-management-plan.md`
   - Or with domain subfolders:
     - `plans/auth/plan-index.md`
     - `plans/auth/login-flow-plan.md`
     - `plans/auth/password-reset-plan.md`
     - `plans/auth/session-management-plan.md`

4. **Cross-Referencing**:
   - When referencing another thought file, use the format: `[[plans/filename]]`
   - Include section references where applicable: `[[plans/filename#section]]`
   - For files in subfolders: `[[plans/domain/filename]]`
   - Always include the `plans/` prefix in references
   - Update cross-references when renaming files

### Implementation Workflow

1. Start with the index file to get an overview
2. Prioritize component files based on dependencies
3. Work through each component file systematically
4. Mark files as complete when finished
5. Update the index file with progress
6. Revisit the index file to ensure overall coherence

## Best Practices

1. **Regular Reviews**: Periodically review all thought files to ensure nothing is overlooked
2. **Consistent Formatting**: Use Markdown formatting consistently across all files
3. **Atomic Thoughts**: Each thought file should address a single coherent topic
4. **Versioning**: Include version numbers for significant revisions
5. **Contextual Links**: Always link to relevant code or documentation
6. **Explicit Dependencies**: Clearly state dependencies between thought files
7. **Actionable Items**: Ensure all plans include concrete, actionable steps
8. **Validation Steps**: Include validation criteria for each major component

By following this enhanced system, I'll maintain a more structured, traceable, and effective approach to problem-solving through thought files.
