from flask import Flask, request, jsonify
from lm_explorer.lm.gpt2 import GPT2LanguageModel
import torch
from time import time
import numpy as np
ar_sents = {}#np.load("/mnt/ed/research_projects/entity_tagger/data/wiki_hyperlink_sentences_short.npy",allow_pickle=True).item()

model_345M = GPT2LanguageModel(model_name='345M',device="cuda")
model_774M = GPT2LanguageModel(model_name='774M',device="cuda")
app = Flask(__name__)

def return_output(next_words, next_probs, sentence, model,sentence_loss,message ):
    return jsonify({
        "probabilities": next_probs,
        "words": next_words,
        "sentence": sentence,
        "message":message,
        "sentence_loss":sentence_loss,
        "model":model
    })

@app.route("/api/get_next_words",methods=['GET'])
def get_next_words():
    input_json = request.get_json()
    if 'sentence' not in input_json:
        return return_output(None,None,None,None,"ERROR! Specify json with 'sentence' parameter")
        
    sentence = input_json["sentence"]
    if len(sentence)>5000:
        return return_output(None,None,None,None,"Sentence longer than 500 characters")
    topk = input_json.get("topk", 10)
    model_name = input_json.get("model", "345M")
    
    start_time = time()
    if model_name =="345M":
        #return return_output(None,None,None,None,None,"ERROR! Model not activated due to limited GPU memory")
        model = model_345M
    elif model_name == "774M":
        model = model_774M
    else:
        return return_output(None,None,None,None,None,"ERROR! Model not available. Available models: [345M, 774M]")
    

    logits = model.predict(previous=sentence)
    sentence_loss = model.get_sentence_loss(sentence)
    probabilities = torch.nn.functional.softmax(logits)
    top_logits, top_indices = logits.topk(topk)
    next_words = [model[idx.item()] for idx in top_indices]
    next_probs = probabilities[top_indices].tolist()
    return return_output(next_words, next_probs, sentence, model_name,sentence_loss,
                        "Successfully processed in {} seconds ".format(round(time()-start_time,4)))

@app.route("/api/get_article_sentences",methods=['GET'])
def get_hyperlinked_sentences():
    input_json = request.get_json()
    article = input_json.get('article')
    if not article:
        return jsonify({'sentences':[],'message':'Please provide a wiki article name'})
    
    num = input_json.get('num',100)
    if article in ar_sents:
        ars = ar_sents[article][:num]
        return jsonify({'sentences':ars,'message':'{} sentences provided'.format(num)})
    else:
        return jsonify({'sentences':[],'message':'wiki article name {} not present'})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=9000,debug=False,threaded=True)