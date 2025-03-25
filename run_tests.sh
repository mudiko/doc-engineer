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

# Run the tests with coverage
poetry run pytest -xvs --cov=. --cov-report=term-missing

echo "✅ Tests completed!" 