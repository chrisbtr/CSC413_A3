# CSC413 Final Project
## Group Name: Boots and Cats
## Members: 
- **Jackson Joseph Hoogenboom**
- **Chris Bostros**
- **Brad Hebert**

# Introduction
Brad

# Model
In this section we will analyze how our model functions for its two use cases and explain how it works to generate music sequences in each case.

## Model Diagram
In this section we will outline the structure of our model for its two use cases, the first use case is generating classical music from scratch, the second use case is where the model is first fed a sample of data and then generates music from using its built up knowledge of this input as some guidance. Below in the first diagram (Figure 1) is the unrolled diagram of our Recurrent Neural Network Model specifically in the case where it generates music from scratch. In the Figure 2 we have our unrolled model for the case of using an input sequence as guidance. 

**Figure 1:** Our Unrolled Generative RNN Model for generating music from scratch.
![Our RNN Model](./model-diagram.png "Our Unrolled Generative RNN model generating from scratch")

**Figure 2:** Our Unrolled Generative RNN Model for generating music using an input sequence as guidance.
![Our RNN Model](./model-diagram-fed-input.png "Our Unrolled Generative RNN model generating from scratch")

From Figure 1 and 2, it is seen that our model is made up two key components the encoder layers, and the decoder layers. The encoder component is made up of the following layers 3 LSTM RNN layers, along with the pitch embedding matrix, and pitch index to one hot vector matrix. On the other side we have the decoder component is made up of the two MLPs for predicting the step and duration of the note from the last encoder layers output, as well as the two pitch decoder layers with the ReLU activation function between the two used for predicting/generating the distribution of notes/pitches to sample from for the next note, which is later fed in as input.

We will now outline how our model functions work to generate the musical sequences, this has two possible use cases, we will start with the use case of feeding the model an input sequence as it encompasses the use case of generating from scratch, and we will mention in our discussion when these two differ.

To begin the case where we feed the model an input sequence we do this one token at a time, as seen in Figure 2, we first feed it Token 1 which is token that consists of the pitch number, which the note we want it to play, as well as the notes step and duration values. We will aso start off by feeding its LSTM hidden states of None since this is the first Token it sees. Once this is input has propogated through all 3 LSTM layers and the hidden states are updated/computed for this time step (the first one) we ignore the last LSTM layers output and feed these hidden state values in as the previous hidden states values into the model along with the next input token made with the next values of same components pitch, step, and duration. We do this to compute the hidden state values for the second time step. We continue this process of feeding in the next token and previous hidden states until we run out of tokens in the input sequence. 

It is at this point where the from scratch prediction would start and where we actually start generating output in the input feed in sequence use case, thes two methods are the same from here on out with the following small difference. The from scratch use case we start this generation portion of functionality off by feeding in the hidden states as None input like we did at the begining of the feeding input sequence use case. In the feeding in input sequence usee case however we instead feed in the hidden states computed at the eend of feeding in the input sequeence as this will contain our encoded memory of the input sequence for which we want to build off of. Moving forward from this, both use casees becomes the same. In both cases once the privious hidden states are fed in we feeed in the beegin seequence token and thhis gets embeded and fed in to the encoding layers so they produce an out to represent the initial note/token distribution. The encoding compoents output is then fed into the decoder layers. Each decoder layer predicts a different potion of the output token two of the MLP decoder layers predict the step and duration of note respectively. The other two layers with a ReLU activation between them generates a dirtubtion from for the pitch/note to be played, this is then smapled from to actually pick the note played. The three compontents are then combined to produce the output, this output token is then fed back into the model as the next input and we feed in the hidden states we just produced as well to then generated the nextoken in the sequence. We repeat this for the desired length specifided by the user and use the output tokens generated to assmeble a sequence which we then compose into a midi file using our data parsing tools. This is how our model works.

## Model Parameter Analysis

Here we analyze the parameters that make up the model. To do this we will go layer by layer. First we have matrix which converts our pitch index into a one hot vector, this is not really a trainable paramter but it is part of the model non the less, and is an identity matrix of size 129x129 as we have a 128 possible notes plus our begin sequence token, and each has a pitch index must convetable to a one hot. Getting into actually learnable parameters we have our pitch embbedding layer (a MLP layer with no bias) it is effectivley a matrix that takes the 129x1 onee hot in a input and extracts the embbedded representation of the pitch which is a continus vector of size 130x1 so this layer gives use 130x129 learnable parameters as it if a fully connected layer. We then concatene this embdeed vecotr with the continous steep and duration from the input on to the end, giving us a 132x1 input vector into the first LSTM RNN encoding layer. This layer gives us 2048x132, 2048x512 plus 2048 * 2 parameters. These parameters come from firstly the matrix used to compute the portion of the i_t, f_t, o_t, anf g_t vectors used in the LSTM computation from the input 132x1 vector, which is where we  get the 2048x132 paramter matrix from. These 2048x512 parameter matrix also come from the computation of the portion of the  i_t, f_t, o_t, anf g_t vectors which come from the 512x1 hidden vector, and lastly the two 2048 parameter vectors come from the bias added to each of these matrix multipliactions. We also note that we have 2048xX here as i_t, f_t, o_t, anf g_t vectors as each is 512x1 (since we have a hidden state size of 512 in these layers) and we compute them by concatenating them ontopof each other which gives a 4*512x1 = 2048x1 vector. The other two embedding layers give use two 2048x512 and two 2048x1 each. This for similar reasons as the firs embedding layer expcet we have two 2048x512 paramter matrices instead as the input is now the same sizee as the hidden since the first encoding LSTM layer outputs a 512x1 veector. Moving on to the Decoder layers these give us the following trainable paramters. Firstly we get two 512x1 and 1x1 parameters from the two fully connected layers that take tehe finaly embdding layers 512x1 output and compute the step and duration respectively. We also get a 512x300 and 300x1 from the first pitch fully connected decoding layer as it takes in 512x1 vector from the last encoding layer and outputs a 300x1 vector to an ReLU activation function passed into the other pitch fully conneected layer. We also have a bias in that layer which explains  a 300x1 vector. Laslty the last pitch decoder layer gives use a 129x300 and 129x1 paramter matrix and vector, as it is a fully connected layer with a bias that takes the 300x1 output vector form the previouc pitch decoder layer and computes the distribtion of the 129 posisble pitch values. All of these learnable paramters sum together to give us 120x129 + 2048x132+2048x512+2048+2048 + 2048x512+2048x512+2048+2048 + 2048x512+2048x512+2048+2048 + 300x512+300 + 129x130+129 + 1x512+1 + 1x512+1 = 5712809

## Model Output Examples

# Data
Chris 
- be sure to explain why we used certain data split

# Training
chris

# Results
Jackson
- how we measure loss and accuracy

# Ethical Considerations
Brad


# Authors
-  Jackson J. Hoogenboom (hoogenb2) 
-  Bradley D. Hebert (hebertbr)
-  Chris Botros (botrosc2)