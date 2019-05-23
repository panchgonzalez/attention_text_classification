#!/bin/sh

# Purpose: rewrite the local branch's commits in order to anonymize
#  author name and email.  Used for blind code reviews.
#
# **NOTE**:  Must be executed on the feature branch for candidate's work
#
# $ git checkout -b candidate/secret-id
# $ git pull interview.bundle first.last
# $ ./anonymize-commits
# $ git push -u origin candidate/secret-id

git filter-branch --force --env-filter '
CORRECT_NAME="Anonymous"
CORRECT_EMAIL="anonymous@example.org"

export GIT_COMMITTER_NAME="$CORRECT_NAME"
export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"

export GIT_AUTHOR_NAME="$CORRECT_NAME"
export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
' -- master..HEAD
