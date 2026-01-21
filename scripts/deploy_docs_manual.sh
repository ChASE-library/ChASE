#!/bin/bash
# Manual documentation deployment script
# Builds docs locally and deploys to GitHub Pages

set -e

# Configuration
GITHUB_REPO="${GITHUB_REPO:-https://github.com/ChASE-library/ChASE.git}"
VERSION_REF="${1:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "HEAD")}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-gh-pages}"

echo "=== Manual Documentation Deployment ==="
echo "Version reference: $VERSION_REF"
echo "Deploy to: $DEPLOY_BRANCH"
echo ""

# Determine version
if [ "$VERSION_REF" = "master" ]; then
    DOC_VERSION="latest"
    VERSION_TAG="latest"
    OUTPUT_DIR="latest"
    IS_TAG=false
else
    # Match both regular versions (v1.7.0) and release candidates (v1.7.0-rc1)
    if [[ "$VERSION_REF" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc[0-9]+)?$ ]]; then
        DOC_VERSION="$VERSION_REF"
        VERSION_TAG="$VERSION_REF"
        OUTPUT_DIR="$VERSION_REF"
        
        # Check if it's a tag (exists in refs/tags)
        if git rev-parse --verify --quiet "refs/tags/$VERSION_REF" >/dev/null 2>&1; then
            IS_TAG=true
            echo "✓ Found tag: $VERSION_REF"
        elif git ls-remote --tags --exit-code "$GITHUB_REPO" "$VERSION_REF" >/dev/null 2>&1; then
            IS_TAG=true
            echo "✓ Found remote tag: $VERSION_REF"
            echo "  Fetching tag from remote..."
            git fetch "$GITHUB_REPO" "refs/tags/$VERSION_REF:refs/tags/$VERSION_REF" 2>/dev/null || true
        elif git rev-parse --verify --quiet "refs/heads/$VERSION_REF" >/dev/null 2>&1; then
            IS_TAG=false
            echo "ℹ Using branch: $VERSION_REF"
            echo "  Note: Tag '$VERSION_REF' does not exist yet"
    else
            IS_TAG=false
            echo "⚠ Warning: '$VERSION_REF' is not a tag or branch"
            echo "  Proceeding with deployment anyway (useful for pre-release documentation)"
        fi
    else
        echo "Warning: '$VERSION_REF' doesn't match version pattern (v*.*.* or v*.*.*-rc*)"
        echo "Deploying as 'latest'"
        DOC_VERSION="latest"
        VERSION_TAG="latest"
        OUTPUT_DIR="latest"
        IS_TAG=false
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
# Fetch the remote branch
    git fetch github $DEPLOY_BRANCH:$DEPLOY_BRANCH 2>/dev/null || true

# Create a temporary branch for deployment
if git show-ref --verify --quiet refs/heads/$DEPLOY_BRANCH; then
    # Branch exists locally, create worktree from it
    git worktree add -f deploy_worktree $DEPLOY_BRANCH 2>/dev/null || true
else
    # Branch doesn't exist locally, create it from remote or as new branch
    if git show-ref --verify --quiet refs/remotes/github/$DEPLOY_BRANCH; then
        git branch $DEPLOY_BRANCH refs/remotes/github/$DEPLOY_BRANCH 2>/dev/null || true
        git worktree add -f deploy_worktree $DEPLOY_BRANCH 2>/dev/null || true
    else
    git worktree add deploy_worktree -b $DEPLOY_BRANCH 2>/dev/null || true
    fi
fi

cd deploy_worktree

# Set up remote tracking and pull latest changes
git remote add github "$GITHUB_REPO" 2>/dev/null || true
git fetch github $DEPLOY_BRANCH 2>/dev/null || true

# Pull/merge remote changes if they exist
if git show-ref --verify --quiet refs/remotes/github/$DEPLOY_BRANCH; then
    echo "Merging remote changes..."
    git merge --no-edit "github/$DEPLOY_BRANCH" 2>/dev/null || {
        echo "⚠️  Merge conflict or merge failed, continuing with local changes..."
    }
fi

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
        echo "   3. Manually resolve conflicts and push"
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
echo "  https://chase-library.github.io/ChASE/$OUTPUT_DIR/"
if [ "$OUTPUT_DIR" != "latest" ]; then
    echo ""
    echo "You can access the documentation at:"
    echo "  https://chase-library.github.io/ChASE/$OUTPUT_DIR/index.html"
fi
echo ""
echo "Note: It may take a few minutes for GitHub Pages to update."

