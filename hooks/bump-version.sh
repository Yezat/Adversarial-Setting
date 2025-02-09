#!/bin/bash

# Run bumpver to update the version
bumpver update --patch

# Stage the version update
git add pyproject.toml

# Amend the last commit with the new version
git commit -m "Bump version to $(bumpver current)"

echo "Version bumped automatically!"
