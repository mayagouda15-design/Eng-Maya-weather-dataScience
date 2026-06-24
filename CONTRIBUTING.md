# Contributing to Weather Data Science Project

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional. We are committed to providing a welcoming and inspiring community for all.

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- pip or conda

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mayagouda15-design/Eng-Maya-weather-dataScience.git
cd Eng-Maya-weather-dataScience

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black

# Make changes and test
python src/analysis.py
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Tests only

### 2. Make Your Changes

Follow these guidelines:
- **Code Style**: PEP 8 compliance
- **Formatting**: Use Black formatter
- **Linting**: Run flake8
- **Comments**: Add docstrings to functions
- **Tests**: Add tests for new functionality

### 3. Format and Lint

```bash
# Format code with Black
black src/

# Check with flake8
flake8 src/

# Run tests
pytest tests/
```

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new forecasting model

- Implement XGBoost integration
- Add model validation tests
- Update documentation
"
```

**Commit message format**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style
- `test:` - Tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

## Pull Request Guidelines

1. **Descriptive Title**: Clear and concise
2. **Detailed Description**: 
   - What changes were made
   - Why the changes were made
   - Any related issues
3. **Screenshots**: If UI changes
4. **Tests**: Include test coverage
5. **Documentation**: Update README if needed

### PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Performance improvement

## Related Issues
Fixes #(issue number)

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] All tests passing

## Documentation
- [ ] README updated
- [ ] Docstrings added
- [ ] Comments added for complex logic
```

## Coding Standards

### Python Style
```python
"""
Module docstring describing the module's purpose.
"""

def function_name(param1: str, param2: int) -> bool:
    """
    Function docstring with description, parameters, and return value.
    
    Args:
        param1: First parameter description
        param2: Second parameter description
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: Description of when this is raised
    """
    pass


class ClassName:
    """Class docstring describing the class."""
    
    def __init__(self, param: str):
        """Initialize the class."""
        self.param = param
```

### File Organization
```python
# Imports (standard library first, then third-party, then local)
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Functions and Classes
def helper_function():
    pass


class MainClass:
    pass
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analysis.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions and classes
- Update AGENTS.md if agent conventions change
- Include examples in docstrings

## Issues and Bug Reports

### Reporting a Bug

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment**
- OS: [e.g. Windows, macOS]
- Python version: [e.g. 3.10]
- Package versions: [e.g. pandas 2.0]

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features.

**Additional context**
Any other context or screenshots.
```

## Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: At least one maintainer reviews
3. **Approval**: Changes are approved
4. **Merge**: Changes are merged to main

## Release Process

1. Update version in `setup.py` or `__version__.py`
2. Update CHANGELOG.md
3. Create release notes
4. Tag release in Git
5. Build and publish package

## Questions or Need Help?

- Check existing issues: https://github.com/mayagouda15-design/Eng-Maya-weather-dataScience/issues
- Create a new issue for questions
- Review documentation in README.md and AGENTS.md

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub insights page

Thank you for contributing! 🙏
