#!/bin/bash

current_dir="$PWD"
parent_dir=$(dirname "$PWD")

sly_repo_dir="${parent_dir}/sly_for_symlink"
# Skip cloning if the directory already exists
if [ -d "$sly_repo_dir" ]; then
    echo "Directory $sly_repo_dir already exists, skipping cloning."
else 
    echo "Directory $sly_repo_dir does not exist, starting to clone the repo."
    mkdir -p "$sly_repo_dir"
    cd "$sly_repo_dir"
    git clone https://github.com/supervisely/supervisely.git
fi

if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found, please execute sh create_venv.sh first."
    exit 1
fi

site_packages_dir=$(find .venv/lib -type d -name "site-packages" -print -quit)

if [ -z "$site_packages_dir" ]; then
    echo "Error: site-packages directory not found."
    exit 1
else
    echo "Site-packages directory found: $site_packages_dir"
fi

if [ -d "${site_packages_dir}/supervisely" ]; then
    rm -r "${site_packages_dir}/supervisely"
fi

cd "$site_packages_dir"
ln -s "${sly_repo_dir}/supervisely/supervisely" .
echo "Symlink created successfully."

echo "Access cloned repository with Supervisely SDK here: $sly_repo_dir"