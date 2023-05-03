Download Link: https://assignmentchef.com/product/solved-cs114-assignment2-neural-transition-based-dependency-parsing
<br>
In this assignment, you’ll be implementing a neural-network based dependency parser, with the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.

<strong>Note</strong>: You are <strong>not </strong>allowed to use any specialized neural network or deep learning libraries, including, but not limited to, CNTK, Keras, MXNet, PyTorch, TensorFlow, Theano, etc. If you have any questions about what libraries you are allowed to use, please ask. It is possible to do the assignment using only those packages imported for you in the starter code.

<h1>Background</h1>

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between <em>head </em>words, and words which modify those heads. Your implementation will be a <em>transition-based </em>parser, which incrementally builds up a parse one step at a time. At every step it maintains a <em>partial parse</em>, which is represented as follows:

A <em>stack </em>of words that are currently being processed.

A <em>buffer </em>of words yet to be processed.

A list of <em>dependencies </em>predicted by the parser.

Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words of the sentence in order. At each step, the parser applies a <em>transition </em>to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied:

SHIFT: removes the first word from the buffer and pushes it onto the stack.

LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack, adding a <em>first word </em>→ <em>second </em><em>word </em>dependency to the dependency list.

RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack, adding a <em>second </em><em>word </em>→ <em>first word </em>dependency to the dependency list.

On each step, your parser will decide among the three transitions using a neural network classifier.

<h1>Assignment</h1>

<ul>

 <li>Go through the sequence of transitions needed for parsing the sentence “<em>I parsed this sentence correctly</em>”. The dependency tree for the sentence is shown below. At each step, give the configuration of the stack and buffer, as well as what transition was applied this step and what new dependency was added (if any). The first three steps are provided below as an example.</li>

</ul>

<table width="639">

 <tbody>

  <tr>

   <td width="132">Stack</td>

   <td width="238">Buffer</td>

   <td width="123">New dependency</td>

   <td width="146">Transition</td>

  </tr>

  <tr>

   <td width="132">[ROOT]</td>

   <td width="238">[I, parsed, this, sentence, correctly]</td>

   <td width="123"></td>

   <td width="146">Initial Configuration</td>

  </tr>

  <tr>

   <td width="132">[ROOT, I]</td>

   <td width="238">[parsed, this, sentence, correctly]</td>

   <td width="123"></td>

   <td width="146">SHIFT</td>

  </tr>

  <tr>

   <td width="132">[ROOT, I, parsed]</td>

   <td width="238">[this, sentence, correctly]</td>

   <td width="123"></td>

   <td width="146">SHIFT</td>

  </tr>

  <tr>

   <td width="132">[ROOT, parsed]</td>

   <td width="238">[this, sentence, correctly]</td>

   <td width="123">parsed → I</td>

   <td width="146">LEFT-ARC</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>A sentence containing <em>n </em>words will be parsed in how many steps (in terms of <em>n</em>)? Briefly explain why.</li>

 <li>Implement the init and parse step functions in the PartialParse class in parser transitions.py. This implements the transition mechanics your parser will use. You can run basic (non-exhaustive) tests by running python parser py part c.</li>

 <li>Our network will predict which transition should be applied next to a partial parse. We could use it to parse a single sentence by applying predicted transitions until the parse is complete. However, neural networks run much more efficiently when making predictions about <em>batches </em>of data at a time (i.e., predicting the next transition for any different partial parses simultaneously). We can parse sentences in minibatches with the following algorithm.</li>

</ul>

<strong>Algorithm 1: </strong>Minibatch Dependency Parsing

<strong>Input : </strong>sentences, a list of sentences to be parsed and model, our model that makes parse decisions

Initialize partial parses as a list of PartialParses, one for each sentence in sentences;

Initialize unfinished parses as a shallow copy of partial parses; <strong>while </strong>unfinished parses is not empty <strong>do</strong>

Take the first batch size parses in unfinished parses as a minibatch;

Use the model to predict the next transition for each partial parse in the minibatch;

Perform a parse step on each partial parse in the minibatch with its predicted transition;

Remove the completed (empty buffer and stack of size 1) parses from unfinished parses;

<strong>end while</strong>

<strong>Return: </strong>The dependencies for each (now completed) parse in partial parses.

Implement this algorithm in the minibatch parse function in parser transitions.py. You can run basic (non-exhaustive) tests by running python parser transitions.py part d.

<strong>Note</strong>: You will need minibatch parse to be correctly implemented to evaluate the model you will build in part (e). However, you do not need it to train the model, so you should be able to complete most of part (e) even if minibatch parse is not implemented yet.

(e) We are now going to train a neural network to predict, given the state of the stack, buffer, and dependencies, which transition should be applied next.

First, the model extracts a feature vector representing the current state. We will be using the feature set presented in the original neural dependency parsing paper: <em>A Fast and Accurate Dependency Parser using Neural Networks</em><a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. The function extracting these features has been implemented for you in utils/parser utils.py. This feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there is one, etc.). They can be represented as a list of integers <strong>w </strong>= [<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>m</sub></em>] where <em>m </em>is the number of features and each 0 ≤ <em>w<sub>i </sub></em>≤ |<em>V </em>| is the index of a token in the vocabulary (|<em>V </em>| is the vocabulary size). Then our network looks up an embedding for each word and concatenates them into a single input vector:

<strong>x </strong>= [<strong>E</strong><em>w</em>1<em>,…,</em><strong>E</strong><em>w</em><em>m</em>] ∈ R<em>dm</em>

where <strong>E </strong>∈ R<sup>|<em>V </em>|×<em>d </em></sup>is an embedding matrix with each row <em>E<sub>w </sub></em>as the vector for a particular word <em>w</em>. We then compute our prediction as:

<strong>h</strong><sub>1 </sub>= ReLU(<strong>xW<sub>1 </sub></strong>+ <strong>b</strong><sub>1</sub>) <strong>h</strong><sub>2 </sub>= ReLU(<strong>h<sub>1</sub>W<sub>2 </sub></strong>+ <strong>b</strong><sub>2</sub>)

<strong>y</strong>ˆ = softmax(<strong>h<sub>2</sub>U </strong>+ <strong>b</strong><sub>3</sub>)

where <strong>h</strong><sub>1 </sub>and <strong>h</strong><sub>2 </sub>are referred to as the hidden layers, <strong>y</strong>ˆ is referred to as the predictions, and ReLU(<em>z</em>) = max(<em>z,</em>0)). We will train the model to minimize cross-entropy loss:

To compute the loss for the training set, we average this <em>J</em>(<em>θ</em>) across all training examples.

We will use UAS score as our evaluation metric. UAS refers to Unlabeled Attachment Score, which is computed as the ratio between the number of correctly predicted dependencies and the number of total dependencies, ignoring the labels (our model doesn’t predict these).

In parser model.py you will find skeleton code to implement this simple neural network using Numpy. Complete the relu, init , embedding lookup, and forward functions to implement the model. Then complete the d relu and train for epoch functions within the run.py file.

Finally execute python run.py to train your model and compute predictions on test data from Penn Treebank (annotated with Universal Dependencies).

<strong>Hints</strong>:

Once you have implemented embedding lookup (e) or forward (f) you can call python parser model.py with flag -e or -f or both to run sanity checks with each function. These sanity checks are fairly basic and passing them doesn’t mean your code is bug free.

When debugging, you can add a debug flag: python run.py -d. This will cause the code to run over a small subset of the data, so that training the model won’t take as long. Make sure to remove the -d flag to run the full model once you are done debugging.

When running with debug mode, you should be able to get a loss smaller than 0.24 and a UAS larger than 65 on the dev set (although in rare cases your results may be lower, there is some randomness when training).

It took about <strong>12 minutes </strong>(using a 2.5 GHz quad-core processor) to train the model on the entire training dataset, i.e., when debug mode is disabled. Your mileage may vary, depending on your computer. That being said, if your model takes hours rather than minutes to train, your code is likely not as efficient as it can be. Make sure your code is fully broadcasted!

When debug mode is disabled, you should be able to get a loss smaller than 0.09 on the train set and an Unlabeled Attachment Score larger than 85 on the dev set. For comparison, the model in the original neural dependency parsing paper gets 92.5 UAS. If you want, you can tweak the hyperparameters for your model (hidden layer size, learning rate, number of epochs, etc.) to improve the performance (but you are not required to do so).

(f) We’d like to look at example dependency parses and understand where parsers like ours might be wrong. For example, in this sentence:

the dependency of the phrase <em>into Afghanistan </em>is wrong, because the phrase should modify <em>sent </em>(as in <em>sent into Afghanistan</em>) not <em>troops </em>(because <em>troops into Afghanistan </em>doesn’t make sense). Here is the correct parse:

More generally, here are four types of parsing error:

<strong>Prepositional Phrase Attachment Error</strong>: In the example above, the phrase <em>into Afghanistan </em>is a prepositional phrase. A Prepositional Phrase Attachment Error is when a prepositional phrase is attached to the wrong head word (in this example, <em>troops </em>is the wrong head word and <em>sent </em>is the correct head word). More examples of prepositional phrases include <em>with a rock</em>, <em>before midnight</em>, and <em>under the carpet</em>.

<strong>Verb Phrase Attachment Error</strong>: In the sentence <em>Leaving the store unattended, I went outside to watch the parade</em>, the phrase <em>leaving the store unattended </em>is a verb phrase. A Verb Phrase Attachment Error is when a verb phrase is attached to the wrong head word (in this example, the correct head word is <em>went</em>).

<strong>Modifier Attachment Error</strong>: In the sentence <em>I am extremely short</em>, the adverb <em>extremely </em>is a modifier of the adjective <em>short</em>. A Modifier Attachment Error is when a modifier is attached to the wrong head word (in this example, the correct head word is <em>short</em>).

<strong>Coordination Attachment Error</strong>: In the sentence <em>Would you like brown rice or garlic naan?</em>, the phrases <em>brown rice </em>and <em>garlic naan </em>are both conjuncts and the word <em>or </em>is the coordinating conjunction. The second conjunct (here <em>garlic naan</em>) should be attached to the first conjunct (here <em>brown rice</em>). A Coordination Attachment Error is when the second conjunct is attached to the wrong head word (in this example, the correct head word is <em>rice</em>). Other coordinating conjunctions include <em>and</em>, <em>but</em>, and <em>so</em>.

In this question are four sentences with dependency parses obtained from a parser. Each sentence has one error, and there is one example of each of the four types above. For each sentence, state the type of error, the incorrect dependency, and the correct dependency. To demonstrate: for the example above, you would write:

<strong>Error type</strong>: Prepositional Phrase Attachment Error

<strong>Incorrect dependency</strong>: troops → Afghanistan <strong>Correct dependency</strong>: sent → Afghanistan

<strong>Note</strong>: There are lots of details and conventions for dependency annotation. If you want to learn more about them, you can look at the UD website: <a href="http://universaldependencies.org/">http://universaldependencies.org</a><a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> or the short introductory slides at: <a href="http://people.cs.georgetown.edu/nschneid/p/UD-for-English.pdf">http://people.cs.georgetown.edu/nschneid/p/ </a><a href="http://people.cs.georgetown.edu/nschneid/p/UD-for-English.pdf">UD-for-English.pdf</a><a href="http://people.cs.georgetown.edu/nschneid/p/UD-for-English.pdf">.</a> However, you <strong>do not </strong>need to know all these details in order to do this question. In each of these cases, we are asking about the attachment of phrases and it should be sufficient to see if they are modifying the correct head. In particular, you <strong>do not </strong>need to look at the labels on the the dependency edges—it suffices to just look at the edges themselves.

ii.

iii.

iv.

<h1>Write-up</h1>

You should also prepare a short write-up that includes at least the following:

The answers to the questions in parts (a), (b), and (f).

The best UAS your model achieves on the dev set and the UAS it achieves on the test set.

<strong>Important</strong>: If you are graduating this semester, please make a note of this, so we know to grade your assignments first!

<a href="#_ftnref1" name="_ftn1">[1]</a> This assignment is adapted from the CS 224N course at Stanford.

<a href="#_ftnref2" name="_ftn2">[2]</a> Chen and Manning, 2014, <a href="https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf">https://nlp.stanford.edu/pubs/emnlp2014-depparser.</a>

<a href="https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf">pdf</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> But note that in the assignment we are actually using UDv1, see: <a href="http://universaldependencies.org/docsv1/">http:// </a><a href="http://universaldependencies.org/docsv1/">universaldependencies.org/docsv1/</a>