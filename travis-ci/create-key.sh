#!/bin/bash

# You may need to `sudo apt-get install ruby2.0 ruby2.0-dev` to get this to work.
if ! $(travis &> /dev/null); then
  sudo gem install travis
fi

ssh-keygen -t rsa -P '' -C "Travis CI deploy key for github.com/google/tensorflow-rust" -f travis_rsa

travis login --org
travis encrypt-file travis_rsa --add

# We don't need the unencrypted private key any more.
rm travis_rsa

git add travis_rsa.enc travis_rsa.pub

echo "Github deploy key is in travis_rsa.pub:"
cat travis_rsa.pub
