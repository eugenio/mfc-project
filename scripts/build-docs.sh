#!/bin/bash

# Build PDF documentation from Markdown files
# Usage: ./build-docs.sh [directory]
PIXI_PROJECT_ROOT=${PIXI_PROJECT_ROOT}
# Set default directory to docs/ if no argument provided
DIR="$PIXI_PROJECT_ROOT/docs"
declare -x PIXI_PROJECT_ROOT
declare -x SVG_FILTER_CACHE_DIR

# Add TeX Live 2025 to PATH for modern emoji support
export PATH="/home/uge/mfc-project/.pixi/envs/docs/texlive2025/2025/bin/x86_64-linux:$PATH"
export TEXMFHOME="/home/uge/mfc-project/.pixi/envs/docs/texlive2025/texmf"
# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "❌ Directory $DIR does not exist"
    exit 1
fi

# Create output directory
mkdir -p "$DIR/pdf"

echo "📄 Building PDF documentation from $DIR..."

# Find all .md files, excluding the pdf subdirectory
find "$DIR" -name "*.md" -not -path "*/pdf/*" | while read -r file; do
    filename=$(basename "$file" .md)
    output_path="$DIR/pdf/$filename.pdf"

    if pandoc --defaults=$DIR/settings.yaml "$file" -o "$output_path"; then
        echo "✅ Successfully built $filename.pdf"
    else
        echo "❌ Failed to build $file"
    fi
done

echo "🎉 Documentation build completed!"