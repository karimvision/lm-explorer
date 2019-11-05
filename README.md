Added some custom modifications from the original repo. This includes,
1. Run model in GPU
2. Custom api to get next words, given a sentence. This is useful for creating custom applications. 

# lm-explorer
interactive explorer for language models (currently only OpenAI GPT-2)

## Running 
```

## Running without Docker

First create and activate a Python 3.6 (or later) virtual environment. Then install the requirements

```bash
$ pip install -r requirements.txt
```

and start the app

```bash
$ python app.py 
```
Make sure you have a ssh tunnel to the server running the app.
You can then access the next words via the following code,

```
import requests
res = requests.get('http://127.0.0.1:9000/api/get_next_words', json={"sentence":"I like to visit", "topk":20})
if res.ok:
    print(res.json())
```

Assuming you have a ssh tunnel from port 8000 to 9000
