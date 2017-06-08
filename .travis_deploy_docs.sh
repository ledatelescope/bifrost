#!/bin/bash
# Credit to https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

# Exit with fail on any errors
set -e

SOURCE_BRANCH="master"
TARGET_BRANCH="gh-pages"

# Pull requests and commits to other branches shouldn't try to deploy, just build to verify
if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "$SOURCE_BRANCH" ]; then
    echo "Skipping deploy; just doing a build."
    exit 0
fi

# Save some useful information
REPO=`git config remote.origin.url`
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
SHA=`git rev-parse --verify HEAD`

# Clone the existing gh-pages for this repo into out/
git clone $REPO out
cd out
git checkout $TARGET_BRANCH
cd ..

# Now build the docs
cd docs
./docker_build_docs.sh # Should put them in html
cp -rf html/* ../out/
cd ../out
./clean_docs.sh # Clean up everything
ls # Doing a dry run.

