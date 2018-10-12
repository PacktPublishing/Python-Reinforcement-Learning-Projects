import tensorflow as tf        #Import dependencies: tensorflow and Numpy
import numpy as np


#Chatbot Based on reinforcement Learning Policy Gradients model
class PolicyGradientDialogue:                            #Policy gradient based Dialogue class
    
    #Initialize
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size,n_steps_encode, n_steps_decode, bias_init = None, learning_rate = 0.00001):
        init = tf.random_uniform_initializer( -0.1, 0.1)        #Random Uniform Initializer for weights 
        self.dim_wordvec = embed_dim                            #Word vector embedding dimension 
        self.dim_hidden = hidden_size                           #Hidden Layer size
        self.n_words = vocab_size                               #Number of words in our vocabulary
        self.n_steps_encode = n_steps_encode                    #Word steps in the LSTM encoder
        self.n_steps_decode = n_steps_decode                    #Word steps in the LSTM decoder
        self.lr = learning_rate                                 #Learning rate
        self.batch_size = batch_size                            #batch size
        
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple = False)   
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple = False)   #LSTM cells for sequential text
        with tf.variable_scope('',reuse = tf.AUTO_REUSE):     #Thesevariables in this scope are reused if they exist or created otherwise
            
            with tf.device("/cpu:0"):         #Use the CPU rather than GPU memory

                self.embed = tf.get_variable('Wemb', [self.n_words, self.dim_hidden], initializer = init)   #Embedding lookup for words in the vocabulary

            self.encode_W = tf.get_variable('encode_vector_W', [self.dim_wordvec, self.dim_hidden], initializer = init)   #Hidden weights on the word vector inputs
            self.encode_b = tf.get_variable('encode_vector_b',[self.dim_hidden], initializer = tf.zeros_initializer ) #Hidden Layer biases 
            self.embed_W = tf.get_variable('embed_word_W', [self.dim_hidden, self.n_words],initializer = init)  #Weights for each word in the vocabulary from the LSMt outputs
            if bias_init is not None:

                self.embed_b = tf.get_variable('embed_word_b',[self.n_words], initializer = tf.constant_initializer( bias_init.astype(np.float32)))
            else:

                self.embed_b = tf.get_variable('embed_word_b',[self.n_words], initializer = tf.constant_initializer( tf.zeros([self.n_words])))  
                #Biases for each word in the word vector based on frequences of the words in the training data
        
        
    
    
    def build_model(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_steps_encode, self.dim_wordvec]) #Inout Placeholder for input sentences in word vector format
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_steps_decode + 1]) #Placeholder for the response to response to the input in the training set with each word represented by its word vector index
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_steps_decode + 1]) 
        #Mask for the caption indication actual words or zero pads
        
        flattened_wordvecs = tf.reshape(word_vectors, [-1, self.dim_wordvec])   #Input word vector is flattened in order to be transformed by the 2D weight matrix
        wordvec_emb = tf.reshape(tf.nn.xw_plus_b(flattened_wordvecs, self.encode_W, self.encode_b), [-1,self.n_steps_encode, self.dim_hidden]) #hidden layer output reshaped back to 3D to be input to the LSTM cells
        
        reward = tf.placeholder(tf.float32, [self.batch_size, self.n_steps_decode]) #Placeholder for the reward at every word/timestep
        
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])       
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        
        padding = tf.zeros([self.batch_size, self.dim_hidden])  #States of the LSMTS are inizialized with zeros
        
        entropies = []
        loss = 0.
        pg_loss = 0.    #policy gradient loss to be minimized 
        
        
        
        for i in range(self.n_steps_encode):   
            
            if i > 0:
                tf.get_variable_scope().reuse_variables()  #Create a new variable at the first time-step and reuse afterwards for all namescopes
            
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:,i,:], state1)  #Run each timestep though the first cell.
                
                
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)
                #Run the output of the first cell through the second cell, the input is concatenated with zeros, the final state after for the responses consists majorly of two components, the latent representation of the the input by the encoder and the state of the deocder based on the selected words. 

                
                
                
        for i in range(self.n_steps_decode): 
            
            with tf.device("/cpu:0"):  #Make use of the CPU for word embeddings
                
                current_embed = tf.nn.embedding_lookup(self.embed, caption[:, i])    #Look up the word embeddings for the word vector for each word in the captio(Response)/
            tf.get_variable_scope().reuse_variables()  #Reuse variables from the encoder
            
            
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)
                
                
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1 ), state2)  #For the other state representation besides those of the encoder.
                
            
            labels = tf.expand_dims(caption[:, i+1], 1) #Expands the dimensions of the the label word to [batch_size,1]
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1),1) #creates a tensor of dimensions[ batch_size, 1] where the entries are numbers from 0 to batch_size -
            labels_onehot = tf.sparse_to_dense(tf.concat([indices, labels],1), tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) #Creates one hot labels of dimensions[batch_size, vocabulary_size] where entries are zero throughout and 1.0 appearing at the timestep.
            
            logit_words = tf.nn.xw_plus_b(output2, self.embed_W, self.embed_b)    #Dense Layer over the output of the LSTM with dimensions[batch_size, number of words] where the magnitude of each entry determines the likelihood of the word at that timestep
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = labels_onehot) * caption_mask[:, i]        #Cross entropy loss between the actual word in the response at the timestep and the prodicted logits. 
            entropies.append(cross_entropy)
            pg_cross_entropy = cross_entropy * reward[:, i]  #Policy gradient loss combining Reinforcement learning rewards(Discussed in another script) with the cross entropy loss
            
            pg_loss += tf.reduce_mean(pg_cross_entropy)  #Mean cross entropy loss across the batch
            
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):      #Adam Optimizer
            train_op = tf.train.AdamOptimizer(self.lr).minimize(pg_loss)
            
            
        input_tensors = {
            'word_vectors' : word_vectors,
            'caption' : caption,
            'caption_mask' : caption_mask,
            'reward': reward }
        
        entropies = {
            'entropies' : entropies }
        return train_op, pg_loss, input_tensors, entropies   #Return inputs placeholder tensors and other tensors such losses and trining optimization operation
    
    
    #Generator function
    def build_generator(self):
          
        #Most of these operations are similar to those in the build_model function and th explanations hold
        
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_steps_encode, self.dim_wordvec])  
        
        flattened_wordvecs = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.reshape(tf.nn.xw_plus_b(flattened_wordvecs, self.encode_W, self.encode_b), [-1,self.n_steps_encode, self.dim_hidden])
        
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        
        padding = tf.zeros([self.batch_size, self.dim_hidden])
                
        generated_words = []
        
        probs = []
        embed = []
        states = []
        
        
        for i in range(0, self.n_steps_encode):  #Same operations as for the build_model encoder
            
            
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:,i,:], state1)
                states.append(state1)
                
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1],1), state2)
                
        for i in range(self.n_steps_decode):
            
            tf.get_variable_scope().reuse_variables()
            
            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.embed, 
                                                           tf.ones([self.batch_size],dtype = tf.int64)) #The beginning of sentence <bos> for the first timestep in the response
            with tf.variable_scope("LSTM1"):
                output_1, state1 = self.lstm1(padding, state1)
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1 ), state2)  #The current words and the encoder state combine to produce the state of the LSTM during decoding
                
            logit_words = tf.nn.xw_plus_b(output2, self.embed_W, self.embed_b)  #Decoder LSTM output fed into a fully collected layer that returns logits representing the probability of each word in the vocabulary
            max_prob_index = tf.argmax(logit_words, 1) #The index of the most probable word corresponding to the maximum logit
            generated_words.append(max_prob_index)     #Append the index of the most probable word to a list for each timestep
            probs.append(logit_words)                  #Append the logits to another list
            
            with tf.device("cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.embed, max_prob_index)  #largest word index is looked up for its word embedding for the decoder at the next time-step
                
            embed.append(current_embed)    #Append the embeddings to another list
            
            
        feats = {
            'probs' : probs,
            'embeds' :embed,
            'states' :states }
        
        return word_vectors, generated_words, feats
                               
                                   
        
        
    