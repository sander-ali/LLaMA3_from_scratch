# LLaMA3_from_scratch
A step-by-step guide to building the complete architecture of the Llama 3 model from scratch and performing training and inferencing on a custom dataset.

Following details are the culmination of information available at 

- Blogs for LLaMA by Meta AI: https://ai.meta.com/blog/meta-llama-3/

- Meta Llama3 Github: https://github.com/meta-llama/llama3

The code implements the architecture in the same sequence as shown in the image below. 

![image](https://github.com/user-attachments/assets/37c0b66b-b995-4243-bdde-54a2156d7a06)

The input block has 3 components Texts/Prompts, Tokenizer, and Embeddings. Therefore, the first step is to code for the input block as shown in the following image

![image](https://github.com/user-attachments/assets/a3b962ad-275f-4a6d-89af-669f5abaf389)

- The input to the model should always be in number format as it is unable to process text. Tokenizer helps to convert these texts/prompts into token-ids (which is an index number representation of tokens in vocabulary). We’ll use the popular Tiny Shakespeare dataset to build the vocabulary and also train our model.
  
- The tokenizer used in the Llama 3 model is TikToken, a type of subword tokenizer. However, we’ll be using a character-level tokenizer for our model building. The main reason is that we should know how to build a vocabulary and tokenizer including encode and decode functions all by ourselves. This way we’ll be able to learn how everything works under the hood and we’ll have full control over the code.
  
- Finally, each token-id will be transformed into an embedding vector of dimensions 128(in original Llama 3 8B, it is 4096). The embeddings will then be passed into the next block called the Decoder Block.

Step 2 is to implement the Decoder Block, which includes:

- RMSNorm
![image](https://github.com/user-attachments/assets/49b4797c-5a3a-4554-aa22-ff6fab160978)

Just like layer normalization, RMSNorm is applied along the embedding features or dimension. The diagram above has embeddings of shape [3,3] meaning each token has 3 dimensions.

Why RMSNorm? RMSNorm reduces the computational overhead by avoiding the calculation of mean and variance. Also, according to the paper by the Author, RMSNorm gives performance advantages while not compromising on accuracy.

- Rotary Positional Encoding
![image](https://github.com/user-attachments/assets/7b78ee86-994a-4488-990d-961bf0e9f589)

![image](https://github.com/user-attachments/assets/d4bdcaa5-0626-4c6c-a370-96e708a678a9)


Let’s say the input text is “I love apple” or “apple love I”, the model will still treat both sentences as the same and learn it as the same. Because there is no order defined in the embeddings for the model to learn. Hence, the order is very important for any language model. In Llama 3 model architecture, RePE is used to define the position of each token in the sentences that maintain not only the order but also maintains the relative position of tokens in the sentences.

RoPE is a type of position encoding that encodes the embeddings which maintains the order of tokens in the sentences by adding absolute positional information as well as incorporates the relative position information among the tokens. It performs the encoding action by rotating a given embedding by a special matrix called the rotation matrix.

Note: the rotation matrix needs to be converted to polar form and the embedding vector needs to converted to complex before performing rotation. After rotation is completed, the rotated embeddings need to be converted back to real for attention operation. Also, RoPE is applied to Query and Key embedding only. It doesn’t apply to Value embedding.

- KV Cache
![image](https://github.com/user-attachments/assets/9283d15a-dad8-478f-832d-ca1776f79647)

In Llama 3 architecture, at the time of inferencing, the concept of KV-Cache is introduced to store previously generated tokens in the form of Key and Value cache. These caches will be used to calculate self-attention to generate the next token. Only key and value tokens are cached whereas query tokens are not cached, hence the term KV Cache.

- Group Query Attention
![image](https://github.com/user-attachments/assets/4306627d-065a-4045-bbc4-1735c316f8f5)

Group query attention is the same as Muilt-Head attention which was used in previous models such as Llama 1 with the only difference being in the use of separate heads for queries and separate heads for keys/values. Usually, the number of heads assigned to queries is n-times to that of keys, and values heads.

The KV cache helps reduce computation resources greatly. However, as KV Cache stores more and more previous tokens, the memory resources will increase significantly. This is not a good thing for the model performance point of view as well as the financial point of view. Hence, Group query attention is introduced. Reducing the number of heads for K and V decreases the number of parameters to be stored, and hence, less memory is being used. Various test results have proven that the model accuracy remains in the same ranges with this approach.

- FeedForward Network (SwiGLU Activation)
![image](https://github.com/user-attachments/assets/7135bab7-c157-4fec-8bc1-b22fdc343e76)

The attention output is first normalized during RMSNorm and then fed into the FeedForward network. Inside the feedforward network, the attention output embeddings will be expanded to the higher dimension throughout its hidden layers and learn more complex features of the tokens.

The SwiGLU function behaves almost like ReLU in the positive axis. However, in the negative axis, SwiGLU outputs some negative values, which might be useful in learning smaller rather than flat 0 in the case of ReLU. Overall, as per the author, the performance with SwiGLU has been better than that with ReLU; hence, it was chosen.

- Decoder Block

- 
The output of the FeedForward network is added again with the attention output. The resulting output is called Decoder output. This decoder output is then fed into another decoder block as input. The same operation is performed for the next 31 decoder blocks. The final decoder output of the 32nd decoder block is then passed to the Output block. 


Finally the output block
![image](https://github.com/user-attachments/assets/46d358fe-4905-4982-be7b-dda0a67a10c7)

The decoder output of the final decoder block will feed into the output block. It is first fed into the RMSNorm. Then, it will feed into the Linear Layer which generates logits. Next, one of the following two operations happens.

If the mode is inference, top_p probability is calculated and the next token is generated. The next tokens generated will stop if the max generation length is reached or the end of sentence token is generated as the next token.

Results after training for 2500 epochs

![image](https://github.com/user-attachments/assets/aea68b05-8394-4e0b-bd38-249c71dfc2ee)


If the mode is Training, loss is computed with the target labels and training is repeated till the max epochs length is reached.

Llama 3 and its other variances are the most popular open-source LLM currently available in the LLM space. I believe the ability to build Llama 3 from scratch provides all the necessary foundation to build a lot of new exciting LLM-based applications. I truly believe that knowledge should be free to all. Feel free to use the source code and update it to build your personal or professional projects. Good luck to you all.
