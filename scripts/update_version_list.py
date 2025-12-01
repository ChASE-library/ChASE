#!/usr/bin/env python3
"""
Script to update the version list in docs/conf.py based on deployed versions.
This can be run as part of the CI pipeline to keep version selector up to date.
"""

import os
import re
import sys

def get_deployed_versions():
    """Get list of deployed versions from GitHub Pages or local directory"""
    versions = ['latest']  # Always include latest
    
    # Check if we're in a CI environment
    if os.environ.get('GITHUB_PAGES'):
        # In GitHub Pages, scan for version directories
        pages_root = os.environ.get('GITHUB_PAGES', '')
        if os.path.exists(pages_root):
            for item in os.listdir(pages_root):
                if os.path.isdir(os.path.join(pages_root, item)):
                    if item.startswith('v') and re.match(r'^v\d+\.\d+\.\d+$', item):
                        versions.append(item)
    
    # Also check from git tags (if available)
    try:
        import subprocess
        result = subprocess.run(['git', 'tag', '--list', 'v*.*.*'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            tags = [tag for tag in result.stdout.strip().split('\n') if tag]
            for tag in sorted(tags, reverse=True):
                if tag not in versions:
                    versions.append(tag)
    except:
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
            versions_dict['latest'] = '/latest/'
        else:
            versions_dict[ver] = f'/{ver}/'
    
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
    conf_path = sys.argv[1] if len(sys.argv) > 1 else 'docs/conf.py'
    
    if not os.path.exists(conf_path):
        print(f"Error: {conf_path} not found")
        sys.exit(1)
    
    versions = get_deployed_versions()
    update_conf_py(conf_path, versions)

