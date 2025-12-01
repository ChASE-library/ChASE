#!/bin/bash
set -e

# Build documentation script for CI
# Detects version from branch name and builds documentation accordingly

echo "=== ChASE Documentation Build Script ==="

# Get version from branch name or environment variable
if [ -n "$CI_COMMIT_REF_NAME" ]; then
    BRANCH_NAME="$CI_COMMIT_REF_NAME"
elif [ -n "$GITHUB_REF" ]; then
    BRANCH_NAME=$(echo "$GITHUB_REF" | sed 's/refs\/heads\///' | sed 's/refs\/tags\///')
else
    BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "master")
fi

# Determine version and output directory
if [ "$BRANCH_NAME" = "master" ]; then
    DOC_VERSION="latest"
    VERSION_TAG="latest"
    echo "Building documentation for: master branch -> latest"
else
    # Extract version from branch name (e.g., v1.3.0, v1.4.0)
    if [[ "$BRANCH_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        DOC_VERSION="$BRANCH_NAME"
        VERSION_TAG="$BRANCH_NAME"
        echo "Building documentation for: $BRANCH_NAME branch -> $VERSION_TAG"
    else
        echo "Warning: Branch '$BRANCH_NAME' does not match version pattern (v*.*.*)"
        echo "Building as 'latest'"
        DOC_VERSION="latest"
        VERSION_TAG="latest"
    fi
fi

# Update Sphinx conf.py with detected version
if [ -f "docs/conf.py" ]; then
    if [ "$VERSION_TAG" != "latest" ]; then
        # Update version in conf.py
        sed -i "s/^version = u'v[0-9.]*'/version = u'$VERSION_TAG'/" docs/conf.py || true
        sed -i "s/^release = u'v[0-9.]*'/release = u'$VERSION_TAG'/" docs/conf.py || true
        echo "Updated docs/conf.py with version: $VERSION_TAG"
    fi
fi

# Create build directory
BUILD_DIR="build_docs"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake .. -DCHASE_BUILD_WITH_DOCS=ON

echo "Building Doxygen documentation..."
make Doxygen

echo "Building Sphinx documentation..."
make Sphinx

# Prepare output directory
OUTPUT_DIR="docs_output"
mkdir -p "$OUTPUT_DIR"

# Copy built documentation
echo "Copying built documentation..."
cp -r sphinx/html/* "$OUTPUT_DIR/"

# Create version info file
echo "$DOC_VERSION" > "$OUTPUT_DIR/.version"
echo "$VERSION_TAG" > "$OUTPUT_DIR/.version_tag"
echo "$BRANCH_NAME" > "$OUTPUT_DIR/.branch"

echo ""
echo "=== Documentation Build Complete ==="
echo "Version: $DOC_VERSION"
echo "Tag: $VERSION_TAG"
echo "Output: $BUILD_DIR/$OUTPUT_DIR/"
echo ""

