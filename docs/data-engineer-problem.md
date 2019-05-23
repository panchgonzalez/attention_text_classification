# Data Engineer Problem

## Overview

Note that both Part 1 & Part 2 of this exercise have open ended elements.
Please make any assumptions necessary for the exercise, and state your assumptions in the `OVERVIEW.md`
file mentioned in the top level README.

We are interested in the choices you make:
- what do you name your scripts?
- how are they invoked?
- how is the output delivered?
- how do you structure your code?
- how do you document your findings and approach?

For both Part 1 & Part 2, your deliverables are:

- source code to achieve the desired results
- A README explaining how to execute your script - make whatever choices you'd like here
- `OVERVIEW.md` populated with a description of your approach
- Any test or other documentation that you believe is necessary

Your solution should be as simple as possible.
Use of outside libraries or resources is expected.

Any technology is welcome, but keep in mind we are a UNIX/Linux development shop
with our tech stack primarily Go, Python, Bash, and JavaScript.

## Part 1:  Data Transformation

For this exercise you will be transforming phone call data from CSV into JSON.
Examples of inputs and outputs are all in the `/data` directory of this repo.
Please note the shape of the CSV data is different from the JSON output.
Note: please resolve any data discrepancies in the best way you can,
documenting your approach in `OVERVIEW.md`.

e.g. When given `calls-4.csv`, your script should output the most equivalent version of `calls-4.json` you can manage.

Example CSV

| call-id | user | start-time | end-time | media-id |
| --- | --- | --- | --- | --- |
| 8808a19c-dd96-462a-9001-6b94ea5df947 | dave911 | 2017-05-31T12:06:00.000Z | 2017-05-31T13:06:00.000Z | dac829e6-b6f9-40ea-8f50-649b99f98865 |
| 4f3535b2-ebef-4f36-90a3-ff646b1d822b | dave072 | 2017-12-16T13:19:00.000Z | 2017-12-16T14:29:00.000Z | 5721221d-370a-4207-a1ba-7fe6fb975e0a |

## Part 2:  Data Validation

For this exercise you will be validating CSV files
and reporting all anomalies you find.

All files without `dirty` in the filename are valid.
These should pass your script with no errors.
We have provided one problematic data file, `data/calls-2k.dirty.csv` containing several errors.
Your script should produce a failing result on this file, and output the list of errors
in the manner of your choosing.

Please provide the output of your validation errors on
`data/calls-2k.dirty.csv` in the `ANSWERS.md` file.

## Part 3:  Data Organization

WARNING:  extremely open-ended question...
To help calibrate how much effort you should spend on this part, we are looking for a page or two of notes in markdown format.
The point is not to tax your time, but to observe how you think and approach problems.

Please describe how you would organize a collection of the JSON files in the examples above.
Consider:

- multiple languages (e.g. English, Spanish, Esperanto)
- multiple industries (e.g. calls in finance vs. 9-1-1 calls)
- records potentially stored for years
- large volumes (peta-bytes) of data

What factors and tradeoffs would you consider?
What questions would you ask before beginning this project?

Please provide your writeup in markdown format in the `ANSWERS.md` file.
