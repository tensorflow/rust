## Releasing

1. Check out a clean copy.  Note that `cargo publish` packages up untracked files.  Use `--allow-dirty` at your peril.
1. Fetch from the main repo
1. Ensure that the TensorFlow version is a real release, not a release candidate
1. Update changelog.
1. Bump version number of `tensorflow-sys` if necessary
   1. Run `git log v${PREVIOUS_VERSION}..HEAD tensorflow-sys` and see if there were any changes. If not, skip.
   1. Bump the version in `tensorflow-sys/Cargo.toml`
   1. Bump the version for `tensorflow-sys` in the root `Cargo.toml`
1. Bump the version number in `Cargo.toml`
1. Bump the version number in `README.md`
1. Run `./test-all`
1. Run `python examples/addition/addition.py` using the version of TensorFlow being linked against.
   (Use pip in a virtualenv, see https://www.tensorflow.org/install/pip#2-create-a-virtual-environment-recommended)
   1. Run `virtualenv --system-site-packages -p python3 ~/tensorflow-${TENSORFLOW_VERSION}`
   1. Run `source ~/tensorflow-${TENSORFLOW_VERSION}/bin/activate` to activate the virtualenv
   1. Run `pip install --upgrade pip`
   1. Run `pip install --upgrade tensorflow==${TENSORFLOW_VERSION}`
   1. Run `python examples/addition/addition.py`
   1. Run `deactivate` to exit the virtualenv
1. Run `./run-valgrind`
1. Commit and push the changes. (Push before publishing to ensure that the changes being published are up to date.)
1. If the version of tensorflow-sys was bumped, run `cargo publish` for tensorflow-sys. (Publish before tagging in case there are problems publishing and we need to add commits to fix them.)
1. Run `cargo publish`. (Publish before tagging in case there are problems publishing and we need to add commits to fix them.)
1. Add a `v${VERSION}` tag and push it
1. Run `./update-docs`
1. Announce the release to the mailing list

## Post-release

1. Update version numbers of dependencies
1. Remove any deprecated items scheduled to be removed
