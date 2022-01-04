# About

This repository contains files from my master's thesis written in 2019. It doesn't contain my previous commit history and all of the files as some of the datasets I used were owned by a private company.

The code is for training, saving and using a write assistant with the following functionalities:
* word prediction
* spelling correction
* correction by using translation from second language (if someone uses a second language and forgets the needed word in the first language)
* spelling correction and correction by using translation (if the second language word is misspelled, it can still give the correct first language word as an offer)

The report can be found here: https://tinyurl.com/kmailla-thesis

Example with English and Danish langauges
![model chart](https://github.com/kmailla/master-thesis/blob/main/model.PNG?raw=true)


# Abstract

Early translation-aware word prediction describes a process where new words are getting suggested to a user based on the text they have already written, with considering the word itself that is currently being typed, even when it is coming from a second language.

The purpose of this thesis is to create the core logic for a write assistant that later can be used as a helper tool by people when they write on their computers. The assistant’s desired functionalities include word suggestions, auto-completion, spelling correction and translation. To be able to realize these goals, the thesis researches ideas coming from the field of natural language processing and deep learning. After getting familiarized with neural networks, the project shows possible designs for a write assistant and points out the main tasks that the implementation can be separated into, namely word prediction, spelling correction, translation and the task of pipelining these together. With each neural network model implementation, an evaluation is made by comparing different variants of the same core model by testing them out on unseen data.

After selecting the best performing models as components, a simple web application is produced for testing the logic and the thesis ends with the conclusion where the ready assistant model is being reviewed. This evaluation shows that the created model is capable of offering words and correcting both first and second language misspellings, and the thesis ends with recommendations on future enhancements.