# GreenKey Interviews

## Overview

The goal of this exercise is to evaluate your thinking and construction skills.
The problem is intended to be small in scope,
but provide a few implementation choices to see how you think.

Feel free to use any languages, tools, or frameworks you like,
although in case it influences your decision the main languages at GreenKey are
JavaScript, Python, Go, and Bash.

We've designed the test to be somewhat related to GreenKey's core business.
We deal with:
- real-time communication
- machine learned transcription
- front-ends to manage and display audio->text
- transforming, cleaning, and managing a large Data Science corpus

This test not only provides a code sample for
our developers and data scientists
to evaluate your skill,
but it will provide a familiar codebase for our in-person interview,
where you will pair with one of our team members to add a feature or two.

Our interview is intended to simulate a real-world work problem,
so use whatever online resources you like.
We typically expect your submission within about a week.

Like a real-world problem, nothing is perfect, including these requirements.
So please ask any clarifying questions (or point out any bugs!)
which help you complete the exercise.
We intentionally leave some things ambiguous.
Feel free to make any assumptions you need,
but please document them in the `OVERVIEW.md` file mentioned below.

## The Problem

### Definitions & Requirements

**Phone Call**  A shared, simultaneous conversation shared by two or more users.  A call may be a group call (i.e. conference call) which has users come and go at any time.

- all start times are inclusive, and end times are exclusive
- all times are to one-second precision only

### Givens

You should have received a `.bundle` file via email, and a link to instructions on
how to use Bundle files.  These [instructions for git bundles](docs/BUNDLE.md) to clone a repository are also included here.

Inside this repository, in the `data/` directory are the data files
you will need for the exercise.

The first row is the column headings, which should be self-explanatory.
Timestamps are formatted according to [RFC 3339](https://tools.ietf.org/html/rfc3339)

From here, please follow the individual questions and requirements for the job description
you're applying for:

- [Back-end Software Developer](docs/backend-problem.md)
- [Customer Engineer](docs/customer-engineer-problem.md)
- [Data Engineer](docs/data-engineer-problem.md)
- [Data Science](docs/data-science-problem.md)
- [DevOps Developer](docs/devops-problem.md)
- [Front-end Software Developer](docs/frontend-problem.md)

## The Solution

### Deliverable

All of the following should be returned as a git repo
packaged as an `interview.bundle` file
as specified in the [Bundle File](docs/BUNDLE.md) help.

- Your code and/or tests
- An `ANSWERS.md` file which contains the answers to any questions asked.
- An `OVERVIEW.md` file which contains:
  - a description of your approach
  - a description of any assumptions you made
  - any instructions required to get your project running
  - any other thoughts or feedback you'd like to share with our team
- A `README.md` file explaining how to execute your script 

Please submit your code with commits like you would
for a normal pull-request on the job.  (i.e. feel free to delete failed avenues)

**NOTE** All submitted commits are anonymized before sharing with the team to maintain an objective process of review. Please avoid placing author notes in the code you submit to maintain anonymity!

Stubs for the markdown files (`ANSWERS.md`, `OVERVIEW.md`) are provided in the top-level directory.

Please email your `interview.bundle` file
back to the person who sent you these instructions.

## Next Steps

After we receive your submission, we will review it as a team.
If we decline to continue the interview process,
we will give you feedback on your code submission with our opinions on how you might improve.
If you pass through to the next step,
we'll bring you in to meet the team and do some pairing on this code submission.
