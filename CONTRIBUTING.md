# Contributing guidelines

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](http://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

Make sure that your [email address in the commit](https://help.github.com/articles/setting-your-commit-email-address-in-git/)
matches the email address you use to sign the CLA, or we won't be able to merge your pull request.
Do this *before* creating the commits.
If you've already created the commits with a different email address, you should be able to sign the CLA again with that email address.

### GitHub Issues

If you want to work on a GitHub issue, check to make sure it's not assigned to someone first.
If it's not assigned to anyone, assign yourself once you start writing code.
(Please don't assign yourself just because you'd like to work on the issue, but only when you actually start.)
This helps avoid duplicate work.

If you start working on an issue but find that you won't be able to finish, please un-assign yourself so other people know the issue is available.
If you assign yourself but aren't making progress, we may assign the issue to someone else.

If you're working on issue 123, please put "Fixes #123" (without quotes) in the commit message below everything else and separated by a blank line.
For example, if issue 123 is a feature request to add foobar, the commit message might look like:
```
Add foobar

Some longer description goes here, if you
want to describe your change in detail.

Fixes #123
```
This will [close the bug once your pull request is merged](https://help.github.com/articles/closing-issues-using-keywords/).

If you're a first-time contributor, try looking for an issue with the label "good first issue", which should be easier for someone unfamiliar with the codebase to work on.

### Git

Please check out a recent version of `master` before starting work, and rebase onto `master` before creating a pull request.
This helps keep the commit graph clean and easy to follow.

As noted in the CLA section, make sure that your [email address in the commit](https://help.github.com/articles/setting-your-commit-email-address-in-git/)
matches the email address you use to sign the CLA, or we won't be able to merge your pull request.
Do this *before* creating the commits.
