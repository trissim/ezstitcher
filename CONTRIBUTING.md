# Contributing to EZStitcher

Thank you for your interest in contributing to EZStitcher! This document provides guidelines for contributing to the project.

## Git Workflow

### Branching Strategy

We follow a simplified version of the GitFlow branching model:

- `main`: The main branch contains the production-ready code. All releases are made from this branch.
- `feature/*`: Feature branches are used for developing new features. They are branched off from `main` and merged back into `main` when the feature is complete.
- `bugfix/*`: Bugfix branches are used for fixing bugs. They are branched off from `main` and merged back into `main` when the fix is complete.

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This leads to more readable messages that are easy to follow when looking through the project history.

Each commit message consists of a **header**, a **body**, and a **footer**. The header has a special format that includes a **type**, a **scope**, and a **subject**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory and the **scope** of the header is optional.

#### Type

The type must be one of the following:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation

#### Scope

The scope should be the name of the module affected (as perceived by the person reading the changelog generated from commit messages).

#### Subject

The subject contains a succinct description of the change:

- Use the imperative, present tense: "change" not "changed" nor "changes"
- Don't capitalize the first letter
- No dot (.) at the end

#### Body

The body should include the motivation for the change and contrast this with previous behavior.

#### Footer

The footer should contain any information about **Breaking Changes** and is also the place to reference GitHub issues that this commit **Closes**.

#### Examples

```
feat(stitcher): add support for Opera Phenix microscope format

Add support for parsing Opera Phenix filenames and directory structure.
This allows the stitcher to work with data from Opera Phenix microscopes.

Closes #123
```

```
fix(zstack): correct Z-stack detection logic

The Z-stack detection logic was incorrectly identifying single-plane images as Z-stacks.
This fix ensures that only true Z-stacks are processed as such.

Closes #456
```

## Pull Requests

- Create a new branch for each feature or bugfix
- Make sure your code follows the project's coding style
- Write tests for your changes
- Make sure all tests pass
- Update documentation if necessary
- Submit a pull request to the `main` branch

## Code Style

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Additionally:

- Use 4 spaces for indentation
- Use docstrings for all modules, classes, and functions
- Use type hints where appropriate
- Keep line length to a maximum of 100 characters

## Testing

- Write unit tests for all new functionality
- Make sure all tests pass before submitting a pull request
- Use pytest for running tests

## Documentation

- Update documentation for all new features and changes
- Use docstrings for all modules, classes, and functions
- Keep documentation up-to-date with the codebase
