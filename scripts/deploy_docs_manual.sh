#!/bin/bash
# Manual documentation deployment script
# Builds docs locally and deploys to GitHub Pages

set -e

# Configuration
GITHUB_REPO="${GITHUB_REPO:-https://github.com/ChASE-library/ChASE.git}"
BRANCH_NAME="${1:-$(git rev-parse --abbrev-ref HEAD)}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-gh-pages}"

echo "=== Manual Documentation Deployment ==="
echo "Branch: $BRANCH_NAME"
echo "Deploy to: $DEPLOY_BRANCH"
echo ""

# Determine version
if [ "$BRANCH_NAME" = "master" ]; then
    DOC_VERSION="latest"
    VERSION_TAG="latest"
    OUTPUT_DIR="latest"
else
    if [[ "$BRANCH_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        DOC_VERSION="$BRANCH_NAME"
        VERSION_TAG="$BRANCH_NAME"
        OUTPUT_DIR="$BRANCH_NAME"
    else
        echo "Warning: Branch '$BRANCH_NAME' doesn't match version pattern"
        echo "Deploying as 'latest'"
        DOC_VERSION="latest"
        VERSION_TAG="latest"
        OUTPUT_DIR="latest"
    fi
fi

echo "Deploying documentation as: $OUTPUT_DIR"
echo ""

# Step 1: Build documentation
echo "Step 1: Building documentation..."
if [ ! -f "scripts/build_docs.sh" ]; then
    echo "❌ Error: scripts/build_docs.sh not found"
    exit 1
fi

chmod +x scripts/build_docs.sh
./scripts/build_docs.sh

if [ ! -d "build_docs/docs_output" ]; then
    echo "❌ Error: Documentation build failed"
    exit 1
fi

echo "✅ Documentation built successfully"
echo ""

# Step 2: Prepare deployment
echo "Step 2: Preparing deployment..."
DEPLOY_DIR="deploy_temp"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy documentation
cp -r build_docs/docs_output "$DEPLOY_DIR/$OUTPUT_DIR"

# Create root redirect
cat > "$DEPLOY_DIR/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=/latest/">
    <link rel="canonical" href="/latest/" />
    <title>ChASE Documentation</title>
</head>
<body>
    <p>Redirecting to <a href="/latest/">latest documentation</a>...</p>
    <script>window.location.replace("/latest/");</script>
</body>
</html>
EOF

echo "✅ Deployment package prepared"
echo ""

# Step 3: Deploy to GitHub Pages
echo "Step 3: Deploying to GitHub Pages..."

# Check if GitHub remote exists
if ! git remote | grep -q "^github$"; then
    echo "Adding GitHub remote..."
    git remote add github "$GITHUB_REPO"
fi

# Checkout or create gh-pages branch
echo "Setting up gh-pages branch..."
if git show-ref --verify --quiet refs/remotes/github/$DEPLOY_BRANCH; then
    echo "Fetching existing gh-pages branch..."
    git fetch github $DEPLOY_BRANCH:$DEPLOY_BRANCH 2>/dev/null || true
fi

# Create a temporary branch for deployment
git worktree add -f deploy_worktree $DEPLOY_BRANCH 2>/dev/null || \
    git worktree add deploy_worktree -b $DEPLOY_BRANCH 2>/dev/null || true

cd deploy_worktree

# Update the version directory
echo "Updating $OUTPUT_DIR directory..."
rm -rf "$OUTPUT_DIR"
cp -r "../$DEPLOY_DIR/$OUTPUT_DIR" .

# Update root index.html
cp "../$DEPLOY_DIR/index.html" .

# Commit and push
git add "$OUTPUT_DIR" index.html
if git diff --staged --quiet; then
    echo "⚠️  No changes to commit"
else
    git commit -m "Update documentation for $OUTPUT_DIR ($(date +%Y-%m-%d))" || echo "No changes"
    echo "Pushing to GitHub..."
    git push github $DEPLOY_BRANCH || {
        echo "❌ Push failed. You may need to:"
        echo "   1. Set up GitHub authentication"
        echo "   2. Check repository permissions"
        exit 1
    }
    echo "✅ Pushed to GitHub"
fi

cd ..
git worktree remove deploy_worktree 2>/dev/null || true

# Cleanup
rm -rf "$DEPLOY_DIR"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Documentation deployed to:"
echo "  https://YOUR_USERNAME.github.io/YOUR_REPO/$OUTPUT_DIR/"
echo ""
echo "Note: It may take a few minutes for GitHub Pages to update."

