#!/bin/bash

# PI-HMARL Test Runner Script
# This script runs all tests and generates coverage reports

set -e  # Exit on error

echo "========================================="
echo "PI-HMARL Test Suite"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo "Consider activating a virtual environment before running tests"
fi

# Check for required dependencies
echo "Checking dependencies..."
python -c "import pytest" 2>/dev/null || {
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Run: pip install pytest pytest-cov"
    exit 1
}

# Create test output directory
TEST_OUTPUT_DIR="test_results"
mkdir -p $TEST_OUTPUT_DIR

# Run unit tests
echo -e "\n${GREEN}Running unit tests...${NC}"
pytest tests/test_suite.py::TestCore -v --tb=short \
    --junitxml=$TEST_OUTPUT_DIR/unit_tests.xml \
    --cov=src --cov-report=html:$TEST_OUTPUT_DIR/coverage_html

# Run model tests
echo -e "\n${GREEN}Running model tests...${NC}"
pytest tests/test_suite.py::TestModels -v --tb=short \
    --junitxml=$TEST_OUTPUT_DIR/model_tests.xml

# Run environment tests
echo -e "\n${GREEN}Running environment tests...${NC}"
pytest tests/test_suite.py::TestEnvironments -v --tb=short \
    --junitxml=$TEST_OUTPUT_DIR/environment_tests.xml

# Run training tests
echo -e "\n${GREEN}Running training tests...${NC}"
pytest tests/test_suite.py::TestTraining -v --tb=short \
    --junitxml=$TEST_OUTPUT_DIR/training_tests.xml

# Run integration tests
echo -e "\n${GREEN}Running integration tests...${NC}"
pytest tests/test_suite.py::TestIntegration -v --tb=short \
    --junitxml=$TEST_OUTPUT_DIR/integration_tests.xml

# Run the full test suite
echo -e "\n${GREEN}Running full test suite with benchmarks...${NC}"
python -m tests.test_suite

# Check code style (optional)
if command -v flake8 &> /dev/null; then
    echo -e "\n${GREEN}Running code style checks...${NC}"
    flake8 src/ --max-line-length=100 --exclude=__pycache__ || {
        echo -e "${YELLOW}Warning: Code style issues found${NC}"
    }
fi

# Check type hints (optional)
if command -v mypy &> /dev/null; then
    echo -e "\n${GREEN}Running type checks...${NC}"
    mypy src/ --ignore-missing-imports || {
        echo -e "${YELLOW}Warning: Type checking issues found${NC}"
    }
fi

# Generate test summary
echo -e "\n${GREEN}Test Summary${NC}"
echo "========================================="
echo "Test results saved to: $TEST_OUTPUT_DIR/"
echo "Coverage report: $TEST_OUTPUT_DIR/coverage_html/index.html"

# Check if all tests passed
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
else
    echo -e "\n${RED}✗ Some tests failed${NC}"
    exit 1
fi