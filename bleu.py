import numpy as np
import nltk

def sample(distr, temp=1.0):
    distr = np.log(distr) / temp
    distr = np.exp(distr) / np.sum(np.exp(distr), axis=1)[:, None]
    return [np.random.choice(MAX_VOCAB, 1, p=distr[b])[0] for b in range(distr.shape[0])]

def test_bleu(model,data_string,n=2):
    """
    Evaluates model's max activation outputs for a real sequence, by BLEU score.
    model is generator, pass in trained
    data_string is list of embedded tokens from real data... [23,244,12,70] etc
    """
    # Get word mapping for outputs -> text
    word_index = load_json_dict('out/word_index.json')
    inv_idx = {v: k for k, v in word_index.items()}

    def toText (list_outputs):
        return [inv_idx[word] for word in list_outputs if word != 0]

    reals = [toText(data_string)]
    result = generate_mimic(model,reals[0])
    fakes = [toText(result)]

    score = nltk.translate.bleu_score.sentence_bleu(reals, fakes)
    print ("Score is:" + str(score))
    

def generate_mimic(generator, truth):
    # Store results in results, but seed with truth
    length = len(truth)
    results = np.zeros(length)

    for i in range(length):
        # Take the last i of truth and feed it in.
        feed = np.append([0 for x in range(0,length-i)],truth[:i])

        distr = generator.predict(feed)
        distr = np.array(distr)
        choice = sample(distr, temp=args.temperature)
        result = choices[0]
        results[i] = result

    return results
