#!/usr/bin/env python3
"""
Script to update the version list in docs/conf.py based on deployed versions.
This can be run as part of the CI pipeline to keep version selector up to date.
"""

import os
import re
import sys

def version_compare(ver1, ver2):
    """Compare two version strings (e.g., 'v1.7.0' vs 'v1.8.0')
    Returns: -1 if ver1 < ver2, 0 if equal, 1 if ver1 > ver2
    """
    # Remove 'v' prefix and split into parts
    v1_parts = [int(x) for x in ver1.lstrip('v').split('.')]
    v2_parts = [int(x) for x in ver2.lstrip('v').split('.')]
    
    # Pad with zeros to same length
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))
    
    for i in range(max_len):
        if v1_parts[i] < v2_parts[i]:
            return -1
        elif v1_parts[i] > v2_parts[i]:
            return 1
    return 0

def get_deployed_versions():
    """Get list of deployed versions from GitHub Pages or local directory"""
    versions = ['latest']  # Always include latest
    min_version = 'v1.5.0'  # Minimum version to include
    
    # Check if we're in a CI environment or have a local gh-pages directory
    pages_root = os.environ.get('GITHUB_PAGES', '')
    
    # Also check common locations for deployed docs
    possible_roots = [
        pages_root,
        'deploy_worktree',  # From deploy_docs_manual.sh
        '../gh-pages',      # If running from scripts directory
        'gh-pages',         # If in root
    ]
    
    for root in possible_roots:
        if root and os.path.exists(root):
            if os.path.isdir(root):
                for item in os.listdir(root):
                    item_path = os.path.join(root, item)
                    if os.path.isdir(item_path):
                        # Match regular versions (v1.7.0) but exclude release candidates
                        if item.startswith('v') and re.match(r'^v\d+\.\d+\.\d+$', item):
                            # Check if version is >= min_version and not a release candidate
                            if version_compare(item, min_version) >= 0:
                                if item not in versions:
                                    versions.append(item)
            break
    
    # Also check remote gh-pages branch for deployed versions
    try:
        import subprocess
        # Check common remote names (github, origin)
        for remote in ['github', 'origin']:
            # Try to fetch gh-pages branch info
            result = subprocess.run(['git', 'ls-remote', '--heads', remote, 'gh-pages'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                # Try to list top-level entries in remote gh-pages branch
                result = subprocess.run(['git', 'ls-tree', '--name-only', f'{remote}/gh-pages'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    entries = [e for e in result.stdout.strip().split('\n') if e]
                    for item in entries:
                        # Match regular versions (v1.7.0) but exclude release candidates
                        # Check if it's a directory by trying to list it (or assume it is if it matches pattern)
                        if item.startswith('v') and re.match(r'^v\d+\.\d+\.\d+$', item):
                            # Check if version is >= min_version
                            if version_compare(item, min_version) >= 0 and item not in versions:
                                versions.append(item)
                break  # Found a remote, no need to check others
    except:
        pass
    
    # Also check from git tags (if available) - but only for versions being built
    # We prioritize deployed folders, so only add tags if they're not already in versions
    # and they're being built for the first time (handled in main function)
    # This section is kept for backward compatibility but won't add versions that aren't deployed
    pass
    
    return sorted(versions, key=lambda x: (x != 'latest', x))

def update_conf_py(conf_path, versions):
    """Update html_context['versions'] in conf.py"""
    with open(conf_path, 'r') as f:
        content = f.read()
    
    # Build versions dictionary
    versions_dict = {}
    for ver in versions:
        if ver == 'latest':
            versions_dict['latest'] = '/ChASE/latest/'
        else:
            versions_dict[ver] = f'/ChASE/{ver}/'
    
    # Find and replace the versions dictionary
    pattern = r"'versions':\s*\{[^}]*\}"
    replacement = "'versions': {\n"
    for ver_name, ver_url in versions_dict.items():
        replacement += f"        '{ver_name}': '{ver_url}',\n"
    replacement += "    }"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(conf_path, 'w') as f:
            f.write(new_content)
        print(f"Updated {conf_path} with versions: {', '.join(versions)}")
        return True
    else:
        print(f"No changes needed in {conf_path}")
        return False

if __name__ == '__main__':
    # Parse arguments: conf_path [current_version_being_built]
    if len(sys.argv) > 1:
        conf_path = sys.argv[1]
        current_version = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        conf_path = 'docs/conf.py'
        current_version = None
    
    if not os.path.exists(conf_path):
        print(f"Error: {conf_path} not found")
        sys.exit(1)
    
    versions = get_deployed_versions()
    
    # Ensure 'latest' is always included (should already be, but double-check)
    if 'latest' not in versions:
        versions.insert(0, 'latest')
        print("Ensuring 'latest' is in version list")
    
    # If a current version is being built, ensure it's included
    if current_version and current_version not in versions:
        # Only add if it matches version pattern and is >= min_version
        min_version = 'v1.5.0'
        if (re.match(r'^v\d+\.\d+\.\d+$', current_version) and 
            version_compare(current_version, min_version) >= 0):
            versions.append(current_version)
            versions = sorted(versions, key=lambda x: (x != 'latest', x))
            print(f"Adding current version being built: {current_version}")
    
    # Final sort to ensure 'latest' is first
    versions = sorted(versions, key=lambda x: (x != 'latest', x))
    
    update_conf_py(conf_path, versions)

