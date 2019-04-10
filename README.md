# dbz-markov-faces

I wanted to see what the buzz with machine learning and AI is.
So I proceeded to use two separate libraries and learn none of those.
Because the libraries encapsulate all the actual hard concepts.

# So What Does This Actually Do

This program will use your camera and draw super saiyan hair for you.

But you also need cheesy, long-winded monologues to really stick it to the villain (or hero).
So it will generate you some DBZ dialogue for you to scream out, you method actor, you.

# Tech

`markovify` is a library that takes in text (called a corpus), find patterns (markov chain), and generate new, emergent text (text generation).

`face_recognition` is a library that has machine learning models that have been trained with faces.
It's scarily accurate, with only one photo as input data.

# Setup

This repo isn't turnkey.
For good reasons.
You need text and I don't want to distribute any copyrighted material.
You need photos and I don't want to give you mine or my friends.

## Corpus

The markov chain requires data, which is up to you to find.
I recommend finding some subtitles of Dragon Ball Z.

Put the files into `sources/sanitized/`.
All files in this dir will be fed in a single, common source of text.
Use the same anime, if you want consistent results.
Or maybe it doesn't matter because most anime is deriviative (braces self with flame shield).

## Photo

Put `.jpeg` photos in `face-data/`.
The name of the file, sans extension, is the name of the person.

# How to run

```sh
python facerec_from_webcam_faster.py
```

Will start the camera and face detections.

```sh
python markove_me.py
```

Will read corpus and generate random sentences.
Hit <CR> to continually generate new lines.
