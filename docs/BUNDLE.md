# GreenKey Interviews : Bundle files

You're here because you're looking at a GreenKey interview packet, and
have this strange `.bundle` file you've been given.

We're going old school on this one.  Git has mechanisms for sharing
repositories between computers that don't share a network.  Commonly,
this was used in email-based workflows.  We're using it today for
sharing a repo with you as a single file and receiving one back the same
way.  It's called a bundle file, usually has the extension `.bundle`,
and is created via `git bundle` (shocking!).  We'll include all the
important instructions here, but if you'd like to learn more:
`man git-bundle`.

## Using the .bundle file ##

### Unpacking ###

First, create a directory for your work:

`$ mkdir <working_dir>`

Then, clone from the bundle file almost like you would from a master repo:

`$ git clone -b master <yourbundle.bundle> <working_dir>`

Now, you can `cd <working_dir>` and see that you're in a normal git repo.

### Working ###

Please do not work on the `master` branch.
Create a branch name with your full name like so:

`git checkout -b <firstname.lastname>`

Don't worry, only the person that imports your work will see your
name, we obfuscate the branch name that reviewers see to minimize any
biases.

Make your commits as normal to solve the given problem, then...

## Packaging up your work ##

When ready to submit, you'll need to bundle your repo up carefully.
Make sure to include the branch name in the command.  This will only
include the branch you specify:

`git bundle create interview.bundle <firstname.lastname>`

Email this `interview.bundle` file back to the address you received
the bundle from in the first place.


## Why all this hassle? ##

Bundle files allow us to share a repo with you without making it
public.  Why a repo?

- **We want to see your commits.** How a candidate structures and
  communicates their work to the team is important to us.
- **We want to review your work.** We push your branch of the
  `.bundle` into our private GitHub interview repository, then create
  a pull request.  This allows our team to review your work using familiar tools.
- **We want to pair with you.** We feel the best interview should
  reflect how we work.  Should you pass through to the next
  step of the interview process, our on-site interview has one session
  of pairing with you to resolve any pull request commentary, just like a
  regular work day.
