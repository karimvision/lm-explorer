from flask import Flask, request, jsonify
from lm_explorer.lm.gpt2 import GPT2LanguageModel
import torch
from time import time


model_345M = GPT2LanguageModel(model_name='345M',device="cuda")
app = Flask(__name__)

def return_output(next_words, next_probs, sentence, message ):
    return jsonify({
        "probabilities": next_probs,
        "words": next_words,
        "sentence": sentence,
        "message":message
    })

@app.route("/api/get_next_words",methods=['GET'])
def get_next_words():
    input_json = request.get_json()
    if 'sentence' not in input_json:
        return return_output(None,None,None,"ERROR! Specify json with 'sentence' parameter")
        
    sentence = input_json["sentence"]
    topk = input_json.get("topk", 10)
    
    start_time = time()
    logits = model_345M.predict(previous=sentence)
    
    probabilities = torch.nn.functional.softmax(logits)
    
    top_logits, top_indices = logits.topk(topk)
    next_words = [model_345M[idx.item()] for idx in top_indices]
    next_probs = probabilities[top_indices].tolist()
    return return_output(next_words, next_probs, sentence,
                        "Successfully processed in {} seconds ".format(round(time()-start_time,4)))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000,debug=False,threaded=True)
