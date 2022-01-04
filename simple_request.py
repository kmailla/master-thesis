# source(s): https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

import requests
import json

# initialize the Keras REST API endpoint URL along with the input
KERAS_REST_API_URL = "http://localhost:5002/predict"

# load the input image and construct the payload for the request
previous_words = "i have"
current_word = "brought"

payload = {"previous_words": previous_words, "current_word": current_word}

# submit the request
r = requests.post(KERAS_REST_API_URL, json=json.dumps(payload))
print(r.text)


