#!/bin/bash
# Run Doc Engineer tests

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Poetry is installed
if ! command_exists poetry; then
  echo "Poetry is not installed. Please run setup.sh first."
  exit 1
fi

echo "🧪 Running Doc Engineer tests..."

# Ensure the lock file is up to date
echo "Updating lock file..."
poetry lock

# Install test dependencies
echo "Installing test dependencies..."
poetry add --group dev pytest pytest-cov || poetry add -D pytest pytest-cov

# Run the tests with coverage
poetry run pytest -xvs --cov=. --cov-report=term-missing

echo "✅ Tests completed!" 