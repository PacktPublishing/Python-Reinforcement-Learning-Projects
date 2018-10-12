import os
import time
import sys
import copy
from data_reader import Data_Reader
import data_parser
from data_parser import refine
import re
from gensim.models import KeyedVectors
from pg_model import PolicyGradientDialogue
from seq_model import Seq2Seq_chatbot
from scipy import spatial
import tensorflow as tf
import numpy as np
import math
from pathlib import Path


model_path = 'model/model-56-3000'
reversed_model_path = 'model/Reversed'
model_name = 'model-56-3000'
reversed_model_name = 'model-63'
start_epoch = 56
start_batch = 0
batch_size = 25
word_count_threshold = 20
r_word_count_threshold = 6
cur_train_index = start_batch*batch_size
max_turns = 10

dull_set = ["I don't know what you're talking about.", "I don't know.", "You don't know.", "You know what I mean.", "I know what you mean.", "You know what I'm saying.", "You don't know anything."]

#Dull responses are some of the generic responses observed in the original seq2seq model which the policy gradients are trained to avoid

dim_wordvec = 300
dim_hidden = 10

n_encode_lstm_step = 22 + 22
n_decode_lstm_step = 22

r_n_encode_lstm_step = 22 
r_n_decode_lstm_step = 22

learning_rate = 0.0001
epochs = 57
batch_size = 1
reversed_batch_size = batch_size
def pad_sequences(sequences, maxlen = n_decode_lstm_step): #Pad with zeros if sequences lenth is less than number of decoder steps or truncate if the length is more
    lengths = []
    dtype = 'int32'
    num_samples = len(sequences)
    x = np.zeros((num_samples, maxlen)).astype(dtype)
    for idx, s in enumerate(sequences):  
        trunc = s[-maxlen:]

        trunc = np.asarray(trunc, dtype=dtype)
        x[idx, :len(trunc)] = trunc
        
    return x

def make_batch_X(batch_X, n_steps_encode, dim_wordvec, word_vector):
    """Returns the world vector representation of the batch input by padding or truncating as may apply with a final dimension of [batch_size, n_steps_encode, word_vector] """
    for i in range(len(batch_X)):
        
        batch_X[i] = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in batch_X[i]]
        if len(batch_X[i]) > n_encode_lstm_step:
            batch_X[i] = batch_X[i][:n_steps_encode]
        else:
            for _ in range(len(batch_X[i]), n_steps_encode):
                batch_X[i].append(np.zeros(dim_wordvec))

    current_feats = np.array(batch_X)
    return current_feats

def make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step):   #process a target batch for training 
    current_captions = batch_Y
    current_captions = list(map(lambda x: '<bos> ' + x, current_captions))  #Append '<bos>' to the beginning of a sentence...
    current_captions = list( map(lambda x: x.replace('.', ''), current_captions))
    current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
    current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
    current_captions = map(lambda x: x.replace('\n', ''), current_captions)
    current_captions = map(lambda x: x.replace('?', ''), current_captions)
    current_captions = map(lambda x: x.replace('!', ''), current_captions)
    current_captions = map(lambda x: x.replace('\\', ''), current_captions)
    current_captions = list(map(lambda x: x.replace('/', ''), current_captions)) #Remove th following symbols from the text

    for idx, each_cap in enumerate(current_captions):
        word = each_cap.lower().split(' ')      
        if len(word) < n_decode_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'#For each sentence in the batch, if the number of words is less than the decode length, append '<eos>' signifying the end of the sentence. Else use only words up to (decoder_length -1) and append with the '<eos>' marker
        else:
            new_word = ''
            for i in range(n_decode_lstm_step-1):
                new_word += word[i] + ' '
            current_captions[idx] = new_word + '<eos>'   

    current_caption_ind = []
    for cap in current_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                current_word_ind.append(wordtoix[word])
            else:
                current_word_ind.append(wordtoix['<unk>'])
        current_caption_ind.append(current_word_ind)   
        """For each caption,the word is fetched with the wordtoix dictionary resulting to a list of size [batch_size], where each entry corresponds to the index of the word in the caption. If word is not contained in the dictionary, '3' is returned which corresponds to the index of '<unk>'."""

        
    current_caption_matrix = pad_sequences(current_caption_ind, maxlen=n_decode_lstm_step)   #Pads the list of caption indices to yield uniform sized captions of length *decode_lstm_step)
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix)))

    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1
        return current_caption_matrix, current_caption_masks
        
""""Current_caption_masks is a matrix of size [batch_size, n_decode_lstm_step + 1] which contains 1 at all entries which contain entries in the caption_matrix and zeros otherwise """

    



def index2sentence(generated_word_index, prob_logit, ixtoword): 
    """ if the predicted word is 'unknown,<unk>--index == 3, replace with the second most probable word
    Also if the predicted word is <pad> representing a pad or <bos> replace with the next most probable word
    """
    for i in range(len(generated_word_index)):
        if generated_word_index[i] == 3 or generated_word_index[i] <= 1:
            sort_prob_logit = sorted(prob_logit[i])
            curindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
            count = 1
            while curindex <= 3:
                curindex = np.where(prob_logit[i] == sort_prob_logit[(-2)-count])[0][0]
                count += 1

            generated_word_index[i] = curindex

    generated_words = []
    for ind in generated_word_index:
        generated_words.append(ixtoword[ind])

    # generate sentence
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1    #The sentence ends where the punctuation <eos> is found The rest of the sentence is truncated
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)

    """ Modify the output sentence to take off '<eos>, <bos>, '--' 
    Start every sentence with a capital letter, replace i, i'm i'd with I, I'm. I'd respectively and end with a full stop '.'
    """
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace('<eos>', '')
    generated_sentence = generated_sentence.replace('--', '')
    generated_sentence = generated_sentence.split('  ')
    for i in range(len(generated_sentence)):
        generated_sentence[i] = generated_sentence[i].strip() 
        if len(generated_sentence[i]) > 1:
            generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.'
        else:
            generated_sentence[i] = generated_sentence[i].upper()
    generated_sentence = ' '.join(generated_sentence)
    generated_sentence = generated_sentence.replace(' i ', ' I ')
    generated_sentence = generated_sentence.replace("i'm", "I'm")
    generated_sentence = generated_sentence.replace("i'd", "I'd")

    return generated_sentence


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def count_rewards(dull_loss, forward_entropy, backward_entropy, forward_target, backward_target):

    forward_entropy = np.array(forward_entropy).reshape(batch_size, n_decode_lstm_step)
    backward_entropy = np.array(backward_entropy).reshape(batch_size, n_decode_lstm_step)
    total_loss = np.zeros([batch_size, n_decode_lstm_step])

    for i in range(batch_size):
            # ease of answering
        total_loss[i, :] += dull_loss[i]
    
            # semantic coherence
        
        forward_len = len(forward_target[i].split())
        backward_len = len(backward_target[i].split())
        if forward_len > 0:
            total_loss[i, :] += (np.sum(forward_entropy[i]) / forward_len)
        if backward_len > 0:
            total_loss[i, :] += (np.sum(backward_entropy[i]) / backward_len)

        total_loss = sigmoid(total_loss) * 1.1

        return total_loss

def train(checkpoint = False):
    global dull_set

    wordtoix, ixtoword, bias_init_vector =       data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
    #ixtoword is a dictionary that returns a word in the vocabulary for every index
    #bias_init_vector returns a bias vector for every word to based on relative frequencies 
    if len(dull_set) > batch_size:
        dull_set = dull_set[:batch_size]      
    else:
        for _ in range(len(dull_set), batch_size):
            dull_set.append('')                     #Truncate the dull set to the batch size else fill with null strings
    dull_matrix, dull_mask = make_batch_Y(
                                batch_Y=dull_set, 
                                wordtoix=wordtoix, 
                                n_decode_lstm_step=n_decode_lstm_step) #Returns the caption matrix and mask for the dull set

    ones_reward = np.ones([batch_size, n_decode_lstm_step])

    g = tf.Graph()       #Policy gradient model graph
    g2 = tf.Graph()      #seq2seq reverse model graph
    default_graph = tf.get_default_graph() 

    with g.as_default():
        model = PolicyGradientDialogue(
                dim_wordvec,dim_hidden,
                len(wordtoix),batch_size,
                n_encode_lstm_step,
                n_decode_lstm_step,
                bias_init_vector,
                learning_rate)
        saver = tf.train.Saver(max_to_keep=100)
        sess = tf.InteractiveSession()
        
        train_op, loss, input_tensors, inter_value = model.build_model()
        tf_states, tf_actions, tf_feats = model.build_generator() 
        tf.global_variables_initializer().run()
        if checkpoint:                  #Use checkpoint to resume from a pretrained state or restart training 
            print("Use Model {}.".format(model_name))
            saver.restore(sess, os.path.join(model_path, model_name))
            print("Model {} restored.".format(model_name))
        else:
            print("Restart training...")
            tf.global_variables_initializer().run()
        
    r_wordtoix, r_ixtoword, r_bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=r_word_count_threshold)
    
    
    """ This module uses the the trained sequence to sequence model to reverse ...A seperate graph and a  separate session """
    with g2.as_default():
        reversed_model = Seq2Seq_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(r_wordtoix),
            dim_hidden=dim_hidden,
            batch_size=reversed_batch_size,
            n_encode_lstm_step=r_n_encode_lstm_step,
            n_decode_lstm_step=r_n_decode_lstm_step,
            bias_init_vector=r_bias_init_vector,
            lr=learning_rate)
        _, _, word_vectors, caption, caption_mask, reverse_inter = reversed_model.build_model()
        sess2 = tf.InteractiveSession()
        saver2 = tf.train.Saver()
        if Path(os.path.join(reversed_model_path, reversed_model_name)).exists():
            saver2.restore(sess2, os.path.join(reversed_model_path, reversed_model_name))
            print("Reversed model {} restored.".format(reversed_model_name))
        else:
            tf.global_variables_initializer().run()

    dr = Data_Reader(cur_train_index=cur_train_index, load_list=False)

    for epoch in range(start_epoch, epochs):  
        n_batch = dr.get_batch_num(batch_size)    #Number of batches
        sb = start_batch if epoch == start_epoch else 0
        for batch in range(sb, n_batch):
            start_time = time.time()

            batch_X, batch_Y, former = dr.generate_training_batch_with_former(batch_size) #generates a batch of the given size from the training data set

            current_feats = make_batch_X(
                            batch_X=copy.deepcopy(batch_X), 
                            n_steps_encode=n_encode_lstm_step, 
                            dim_wordvec=dim_wordvec,
                            word_vector=word_vector)

            current_caption_matrix, current_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(batch_Y), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)
            

            """Generate actions(responses) for the state of the dialogue"""
            action_word_indexs, inference_feats = sess.run([tf_actions, tf_feats],
                                                                feed_dict={
                                                                   tf_states: current_feats
                                                                })
            action_word_indexs = np.array(action_word_indexs).reshape(batch_size, n_decode_lstm_step) #Predicted words at each time step
            action_probs = np.array(inference_feats['probs']).reshape(batch_size, n_decode_lstm_step, -1) #The logits representing the relative probabilities for each word in the vocabulary

            actions = []
            actions_list = []
            for i in range(len(action_word_indexs)):
                action = index2sentence(
                                generated_word_index=action_word_indexs[i], 
                                prob_logit=action_probs[i],
                                ixtoword=ixtoword)
                actions.append(action)           #Generates a coherent sentence from the predicted words and stacks them for each state in the batch
                actions_list.append(action.split())     

                action_feats = make_batch_X(
                                batch_X=copy.deepcopy(actions_list), 
                                n_steps_encode=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)   #Returns a an input batch from the previous output

                action_caption_matrix, action_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(actions), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step) #Returns a target(caption_matrix and mask ) from the previous output

                    
                """
                The ease of answering loss partains to the probilility of the response generating any of the responses fromt the dull set. This helps to prevent one of the primary problems in the original seq2seq model where the dialogue dulls out to generic responses. The actions generated are fed as input and the negative of the cross entropy loss with the dull set is take. hence a greater loss is incurred when the response is likely to yield such dull responses. This makes it possible for the agent to genertate lenghtier dialogues
                """
            dull_loss = []
            for vector in action_feats:
                action_batch_X = np.array([vector for _ in range(batch_size)])
                d_loss = sess.run(loss, feed_dict={
                                    input_tensors['word_vectors']: action_batch_X,
                                    input_tensors['caption']: dull_matrix,
                                    input_tensors['caption_mask']: dull_mask,
                                    input_tensors['reward']: ones_reward
                                })
                d_loss *=  -1. / len(dull_set)
                dull_loss.append(d_loss)
                
                    



                # semantic coherence
                """measures the adequacy of responses to avoid situations in which the generated replies are highly rewarded but are ungrammatical or not coherent. We therefore consider the mutual information between the action a and previous turns in the history to ensure the generated responses are coherent and appropriate...This combines the forward entropy and the backward entropy"""
                
                
                #Forward entropy between the dialogue input and the response
                forward_inter = sess.run(inter_value,feed_dict={input_tensors['word_vectors']: current_feats, input_tensors['caption']: action_caption_matrix,input_tensors['caption_mask']: action_caption_masks,input_tensors['reward']: ones_reward })
                                    
                                    
                                    
                                    
                forward_entropies = forward_inter['entropies']
                former_caption_matrix, former_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(former), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)
                
                action_feats = make_batch_X(
                                batch_X=copy.deepcopy(actions_list), 
                                n_steps_encode=r_n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)
               #Backward entropy loss measures the cross between the action and it's input when fed in a reverse seq2seq model 
                backward_inter = sess2.run(reverse_inter,
                                 feed_dict={
                                    word_vectors: action_feats,
                                    caption: former_caption_matrix,
                                    caption_mask: former_caption_masks
                                })
                backward_entropies = backward_inter['entropies']

                # reward: count goodness of actions
                rewards = count_rewards(dull_loss, forward_entropies, backward_entropies, actions, former)
    
                # policy gradient: train batch with rewards
                if batch % 10 == 0:
                    _, loss_val = sess.run(
                            [train_op, loss],
                            feed_dict={
                                input_tensors['word_vectors']: current_feats,
                                input_tensors['caption']: current_caption_matrix,
                                input_tensors['caption_mask']: current_caption_masks,
                                input_tensors['reward']: rewards
                            })
                    print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
                else:
                    _ = sess.run(train_op,
                                 feed_dict={
                                    input_tensors['word_vectors']: current_feats,
                                    input_tensors['caption']: current_caption_matrix,
                                    input_tensors['caption_mask']: current_caption_masks,
                                    input_tensors['reward']: rewards
                                })
                if batch % 1000 == 0 and batch != 0:
                    print("Epoch {} batch {} is done. Saving the model ...".format(epoch, batch))
                    saver.save(sess, os.path.join(model_path, 'model-{}-{}'.format(epoch, batch)))

        print("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
