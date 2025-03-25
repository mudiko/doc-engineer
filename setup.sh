#!/bin/bash
# Doc Engineer setup script

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "🔧 Setting up Doc Engineer..."

# Check if Poetry is installed
if ! command_exists poetry; then
  echo "Poetry not found. Would you like to install it? (y/n)"
  read -r install_poetry
  if [[ "$install_poetry" =~ ^[Yy]$ ]]; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for the current session
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "Poetry installation skipped. You'll need to install dependencies manually."
    exit 1
  fi
fi

# Generate lock file
echo "Generating lock file..."
poetry lock

# Install dependencies with Poetry
echo "Installing dependencies..."
poetry install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file..."
  echo "# Add your API keys below" > .env
  echo "GOOGLE_API_KEY=" >> .env
  echo ".env file created. Please edit it to add your API key."
fi

# Make the doc_engineer.py executable
chmod +x doc_engineer.py

echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "1. Edit the .env file to add your Google API key"
echo "2. Run 'poetry shell' to activate the virtual environment"
echo "3. Generate your first document: ./doc_engineer.py \"Your Document Title\""
echo ""
echo "Or use without activating the shell:"
echo "poetry run doc-engineer \"Your Document Title\"" 