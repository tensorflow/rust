#!/bin/bash

set -ev

[ ! -z "${RUSTDOC_VERSION}" ] && [ "${TRAVIS_RUST_VERSION}" != "${RUSTDOC_VERSION}" ] && exit
[ -z "${RUSTDOC_VERSION}" ] && [ "${TRAVIS_RUST_VERSION}" != "nightly" ] && exit
[ "${TRAVIS_BRANCH}" != "master" ] && exit

git config user.name "Travis CI"
git config user.email ""

if [ -z "${CRATE_NAME}" ]; then
  CRATE_NAME=$(echo ${TRAVIS_REPO_SLUG} | cut -d '/' -f 2 | sed 's/-/_/g')
fi

URL="tensorflow/index.html"

HTML="<!DOCTYPE html>
<link rel='canonical' href='${URL}'>
<meta http-equiv='refresh' content='0; url=${URL}'>
<script>window.location='${URL}'</script>"

echo "${HTML}" > target/doc/index.html

export PYTHONUSERBASE="${HOME}/.local"
pip install ghp-import --user ${USER}
${PYTHONUSERBASE}/bin/ghp-import -m 'Update the documentation' -n target/doc

git push -qf git@github.com:${TRAVIS_REPO_SLUG}.git gh-pages
