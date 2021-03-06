Added some custom modifications to the original repo. This includes,
1. Run model in GPU
2. API to get next words, given a sentence. This is useful for creating custom applications. 

# lm-explorer
interactive explorer for language models (currently only OpenAI GPT-2)

## Running
First create and activate a Python 3.6 (or higher) virtual environment. Then install the requirements

```
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
