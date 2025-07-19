# Contributing to PI-HMARL

Thank you for your interest in contributing to PI-HMARL! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Pull Request Process](#pull-request-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

1. Python 3.8 or higher
2. Git
3. A GitHub account

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Visit https://github.com/your-org/pi-hmarl
   # Click "Fork" button
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pi-hmarl.git
   cd pi-hmarl
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. **Install in development mode**
   ```bash
   pip install -e .
   ```

6. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Process

### 1. Find an Issue

- Check the [issue tracker](https://github.com/your-org/pi-hmarl/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clean, well-documented code
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test
pytest tests/test_suite.py::TestCore -v

# Check code style
flake8 src/
black src/ --check

# Run type checking
mypy src/
```

### 5. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature description"
# or
git commit -m "fix: resolve issue #123"
```

Follow conventional commit format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` test additions/changes
- `chore:` maintenance tasks

## Pull Request Process

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

3. **PR Requirements**
   - Clear description of changes
   - Reference related issues
   - All tests passing
   - Code review approval
   - Documentation updated

4. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Related Issue
   Fixes #(issue number)

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good example
class MultiAgentSystem:
    """Multi-agent system implementation.
    
    Args:
        config: Configuration dictionary
        
    Attributes:
        agents: List of agents
        communication_network: Communication network
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = []
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize agents based on configuration."""
        for i in range(self.config['num_agents']):
            agent = Agent(f"agent_{i}")
            self.agents.append(agent)
```

### Key Guidelines

1. **Line Length**: Maximum 100 characters
2. **Imports**: Group and sort imports
3. **Type Hints**: Use type hints for function arguments and returns
4. **Docstrings**: Use Google-style docstrings
5. **Private Methods**: Prefix with underscore
6. **Constants**: Use UPPER_CASE

### Code Organization

```
src/
├── core/           # Core functionality
├── models/         # Neural network models
├── environments/   # Environment implementations
├── training/       # Training algorithms
├── optimization/   # Performance optimization
├── security/       # Security features
├── hardware/       # Hardware interfaces
├── scenarios/      # Pre-built scenarios
└── utils/         # Utility functions
```

## Testing Guidelines

### Writing Tests

```python
import unittest
import numpy as np
from src.core import MultiAgentSystem

class TestMultiAgentSystem(unittest.TestCase):
    """Test cases for MultiAgentSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'num_agents': 4,
            'state_dim': 12,
            'action_dim': 4
        }
        self.mas = MultiAgentSystem(self.config)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertEqual(len(self.mas.agents), 4)
        self.assertIsNotNone(self.mas.communication_network)
    
    def test_agent_communication(self):
        """Test inter-agent communication."""
        message = {'type': 'test', 'data': 'hello'}
        self.mas.agents[0].send_message(message)
        
        received = self.mas.agents[1].receive_messages()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]['data'], 'hello')
```

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark performance
4. **Hardware Tests**: Test hardware interfaces (when available)

### Coverage Requirements

- Aim for >80% code coverage
- Critical paths must have 100% coverage
- Use `pytest-cov` to measure coverage

## Documentation

### Code Documentation

```python
def compute_physics_loss(
    self,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor
) -> torch.Tensor:
    """Compute physics-informed loss.
    
    Args:
        states: Current states [batch_size, state_dim]
        actions: Actions taken [batch_size, action_dim]
        next_states: Resulting states [batch_size, state_dim]
        
    Returns:
        Physics violation loss
        
    Raises:
        ValueError: If tensor dimensions don't match
    """
    # Implementation
```

### Documentation Types

1. **API Documentation**: Docstrings for all public APIs
2. **User Guide**: How to use the framework
3. **Developer Guide**: How to extend the framework
4. **Examples**: Working examples and tutorials

### Building Documentation

```bash
cd docs/
make html
# View at docs/_build/html/index.html
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussions
- **Discord**: Real-time chat (invite link in README)
- **Email**: pi-hmarl@your-org.com

### Getting Help

1. Check existing documentation
2. Search closed issues
3. Ask in Discord #help channel
4. Create a new issue

### Recognition

We recognize contributors in several ways:
- Contributors list in README
- Credit in release notes
- Community spotlight blog posts

## Additional Resources

- [Development Roadmap](https://github.com/your-org/pi-hmarl/projects)
- [Architecture Overview](docs/architecture.md)
- [API Reference](https://pi-hmarl.readthedocs.io)
- [Research Papers](docs/papers.md)

## Questions?

If you have questions about contributing:
1. Check this guide
2. Ask in Discord #contributors channel
3. Email maintainers at pi-hmarl@your-org.com

Thank you for contributing to PI-HMARL!