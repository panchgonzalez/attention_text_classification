# Back-end Software Developer Problem

## Part 1:  Base Questions

Using data file `calls-10k.csv`, model these calls as you choose.
Please provide code which can be used in a simple way to answer the following questions.

(Keep in mind we will be extending this exercise for your in-person interview)

It is recommended that you provide a driver (which could be a test suite)
which exercises your code to produce these answers.

1. User with most calls
   1. Which user is the most common caller by number of calls?
   1. How much total time did the most common caller spend on calls?
1. Call with most users
   1. Which conference call had the most number of peak callers?
      As callers may come and go, this is defined as the max number of simultaneous users throughout a call (i.e. for same `call-id`)
   1. What is the time range for this peak?
      (i.e. from what start time to end time are the max users on the call)

For your testing purposes, answers to the dataset in `calls-4.csv` are:

```
Question 1a:  eve386
Question 1b:  03:02:00
Question 2a:  #uuid "493abc93-e425-4d53-82a7-aa00710cf5fb"
Question 2b:  [2017-03-16T22:41:00.000Z 2017-03-16T23:55:00.000Z]
```

## Part 2:  Advanced Question(s)

WARNING:  extremely open-ended question...
We are looking for a maximum of a couple pages of markdown here.
The point is not to tax your time, but to understand how you think and approach problems.

Besides the base questions above, please describe how you would solve this problem
if the dataset was too large, by two orders of magnitude, to load into the memory of your machine.  From an algorithmic perspective, what approach would you take?  What factors and tradeoffs would you consider?

Please provide your writeup as markdown in the `ANSWERS.md` file.
