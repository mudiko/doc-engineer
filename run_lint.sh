#!/bin/bash
# Run Doc Engineer linters

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Poetry is installed
if ! command_exists poetry; then
  echo "Poetry is not installed. Please run setup.sh first."
  exit 1
fi

echo "🔍 Running linters on Doc Engineer..."

# Ensure the lock file is up to date
echo "Updating lock file..."
poetry lock

# Make sure we have the lint tools installed
echo "Installing linting tools..."
poetry add --group dev flake8 black mypy || poetry add -D flake8 black mypy

# Run flake8
echo "Running flake8..."
poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
poetry run flake8 . --count --max-complexity=10 --max-line-length=120 --statistics

# Run black
echo "Running black..."
poetry run black --check .

# Run mypy
echo "Running mypy..."
poetry run mypy --ignore-missing-imports .

echo "✅ Lint completed!"

# Offer to fix formatting issues automatically
read -p "Would you like to automatically fix formatting issues with black? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  poetry run black .
  echo "Formatting issues fixed."
fi 