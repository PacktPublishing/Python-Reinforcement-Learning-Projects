import re
import os
import time
import sys
import data_parser
from gensim.models import KeyedVectors
from pg_model import PolicyGradientDialogue
import tensorflow as tf
import numpy as np
from convert_checkpoint import convert_checkpoint

default_model_path = 'model/model-56-3000/model-56-3000'   #Path to Trained model
testing_data_path = 'results/sample_input.txt'    
output_path = 'results/sample_output_RL.txt'
word_count_threshold = 20
dim_wordvec = 300
dim_hidden = 1000
n_encode_lstm_step = 22 + 1 # one random normal as the first timestep to producr variation in responses
n_decode_lstm_step = 22 
batch_size = 1


def test(model_path=default_model_path):
    if model_path == default_model_path:
        model_path = convert_checkpoint(model_path)
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    testing_data = open(testing_data_path, 'r').read().split('\n')   #Contains preset questions to generate responses to

    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)  #Load word_vector which returns a unique vector of dim_wordvec for every word in the corpus

    _, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    #ixtoword is a dictionary that returns a word in the vocabulary for every index
    #bias_init_vector returns a bias vector for every word to based on relative frequencies 
    model = PolicyGradientDialogue(
            embed_dim=dim_wordvec,
            vocab_size=len(ixtoword),
            hidden_size=dim_hidden,
            batch_size=batch_size,
            n_steps_encode=n_encode_lstm_step,
            n_steps_decode=n_decode_lstm_step,
            bias_init=bias_init_vector)
    
    
    word_vectors, caption_tf, feats = model.build_generator()
    
    saver = tf.train.Saver() 
    print('\n=== Use model', model_path, '===\n')
    
    saver.restore(sess,model_path)
    print("Model restored")
    
    with open(output_path, 'w') as out:
        generated_sentences = []
        for idx, question in enumerate(testing_data):
            print('question =>', question)

            question = [data_parser.refine(w) for w in question.lower().split()]   #Converts to lower case and extracts the only the vocabulary parts 
            question = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in question] #Return  the word vector representation for each word in the sentence
            question.insert(0, np.random.normal(size=(dim_wordvec,))) # insert random normal word vector at the first step

            if len(question) > n_encode_lstm_step:
                question = question[:n_encode_lstm_step]   #If quesstion is longer than the encoder, truncate to encoder size
            else:
                for _ in range(len(question), n_encode_lstm_step):   #If less, pad with zeros 
                    question.append(np.zeros(dim_wordvec))

            question = np.array([question]) 
    
            generated_word_index, prob_logit = sess.run([caption_tf, feats['probs']], feed_dict={word_vectors: question})     #Runs the tensorflow session
               #generated_word_index returns the index of the most probable word
               #prob_logit returns the logits for every word at each timestep
            generated_word_index = np.array(generated_word_index).reshape(batch_size, n_decode_lstm_step)[0]
            prob_logit = np.array(prob_logit).reshape(batch_size, n_decode_lstm_step, -1)[0]
            for i in range(n_decode_lstm_step):
                if generated_word_index[i] == 3:    #word is <unk>    
                    sort_prob_logit = sorted(prob_logit[i])
                    
                    maxindex = np.where(prob_logit[i] == sort_prob_logit[-1])[0][0]
                    secmaxindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
                    
                    generated_word_index[i] = secmaxindex      #Returns the second most probable index if the most probable is 3 which corresponds to unknown '<unk>'      

            generated_words = []
            for ind in generated_word_index:
                generated_words.append(ixtoword[ind])    #Returns the word for each index

            # generate sentence
            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation]    #Truncates all words following the '<eos>'---End of sentence marker
            generated_sentence = ' '.join(generated_words)

            # modify the output sentence 
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')
            generated_sentence = generated_sentence.replace('--', '') #Exclude <bos>, <eos> and '--'
            generated_sentence = generated_sentence.split('  ') # Split double spaces into seperate sentences
            for i in range(len(generated_sentence)):
                generated_sentence[i] = generated_sentence[i].strip()
                if len(generated_sentence[i]) > 1:
                    generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.'
                else:
                    generated_sentence[i] = generated_sentence[i].upper()  #starts a sentence with uppercase and ends with a fullstop    
            generated_sentence = ' '.join(generated_sentence)
            generated_sentence = generated_sentence.replace(' i ', ' I ')
            generated_sentence = generated_sentence.replace("i'm", "I'm")
            generated_sentence = generated_sentence.replace("i'd", "I'd")
            generated_sentence = generated_sentence.replace("i'll", "I'll")
            generated_sentence = generated_sentence.replace("i'v", "I'v")
            generated_sentence = generated_sentence.replace(" - ", "")

            print('generated_sentence =>', generated_sentence) #Carry out the above replacements and print
            out.write(generated_sentence + '\n')

