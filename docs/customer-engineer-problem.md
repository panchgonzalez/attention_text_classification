# Customer Engineer Problem

As a Customer Engineer at GreenKey, you will be tasked with helping our clients deploy our APIs in their environments and navigate them through configuration and customization of our software. This assessment covers some of the basic technical skills necessary for success in this role.

When completing this assessment, feel free to use any outside resource, but only write code that you feel comfortable explaining.

## Part 1: Deploying and Using GreenKey Scribe’s Transcription API

1. Setup your environment
    - GreenKey’s APIs for voice and transcription are all containerized using Docker. You will need access to a machine that can run Docker to complete this assessment. Install the community edition of Docker on your machine using the instructions here: https://docs.docker.com/install/
    - If you do not have access to a machine that can run Docker, contact us.


2. Download the GreenKey Scribe Docker container
    - Follow the instructions here to setup Scribe on your machine: https://transcription.greenkeytech.com/svt-e0286da/
    - You can use the following credentials to get started:
      - Repository username: gktuser
      - Repository password: ac12dba5989f5e91bda1e213204d6bb7be3c36e4
     - scribe-username: interview
     - scribe-secretkey: <sent separately>
   
3. Launch a transcription service and transcribe a file
    - Using any language of your choice, make a POST request to transcribe the following file with the following options enabled: https://storage.googleapis.com/gk-transcription.appspot.com/testing/comcast.wav
    - Options enabled:
      - Diarization on
      - Gender identification on
      - Insights on
    - Documentation on options: https://transcription.greenkeytech.com/svt-e0286da/#configuring-scribe
    - A Python example script is presented in our SDK documentation. You can use this as a resource whether or not your script is written in Python: https://transcription.greenkeytech.com/svt-e0286da/sdk/

4. Parse the resulting JSON output
    - Using any language of your choice, parse the JSON response from transcription of the file into a text file that lists the transcription segments in the following format: `startTimeSec – stopTimeSec : gender : speakerID : transcript`
      - Note that `speakerID` refers to the nested ID under the speakerInfo key.

5. Putting it all together
    - Send us the following:
      1. Your Docker run command for launching a transcription service
      2. Your code for making a POST request to the service and parsing the JSON output
      3. The parsed text output
   
## Part 2: Providing Feedback

Combine your responses for both of the questions below into a single text file.

1. Customer Interactions
    - One of GreenKey’s clients send you the following email. Prepare a response with suggestions on how they can address their issues. Note, you will not be able to post to the transcription service using the LICENSE_KEY provided by the customer, but you should be able to successfully execute the `run` the command with a few modifications:
      > Hey everyone, We were able to successfully download GreenKey’s Scribe container. However, we tried launching using the command below and are seeing error messages related to illegal characters and unknown flags. Any suggestions on how to resolve this? Thanks, Jim

          # docker run --d \
          > -p 5000:5000 \
          > -e
          LICENSE_KEY="//AiKLRX+33X5CXqeG7M6kA863PbEmc5NOtpFbno9ppV
          meflIpxKIw11mVj2nJvQ89VTwpv3OM8J9EIt3j2TsY8RIHr8qg+IEHK9Z
          wyF7io=" \
          > -e PROCS="4" \
          > -e KEEP_ALIVE=”True” \
          docker.greenkeytech.com/svtserver:latest

2. Knowledge Base

Please create a sample entry in a knowledge base, describing the customer problem and your solution.  The goal is to build up a body of questions and answers to allow rapid responses from your colleagues to common problems.  Use whatever formatting and language you feel appropriate.

3. Internal Feedback

API documents are constantly evolving as new features and use cases arise. Give us some feedback on the documentation you’ve read so far. What are the best aspects? What could be improved? Summarize your response as if you were emailing someone else at the company with these suggestions.
