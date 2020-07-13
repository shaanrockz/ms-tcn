echo "Running black..."
black -l 80 .

echo "Running flake..."
flake8 .