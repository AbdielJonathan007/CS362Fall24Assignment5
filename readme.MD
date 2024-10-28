
Assignment 5: Managing uncertainty.

##### Due date: Mon Nov 11, 11:59pm

submission.py. No submission.py needed for this assignment! The
code portions should be able to run from the command line. Please 
include a single PDF with the written answers.

####  Problem 1: Decision Trees in scikit-learn. 

Now that you've implemented the decision tree yourself, 
you'll see how to do it in scikit-learn, and how to extend it to an 
ensemble-based approach called Random Forest.

First, take a look at the first set of code in sklearn_decisiontree.py.
This shows how to use the KFold object, along with 
the sklearn decision tree, to *fit* and *score* a model. 

**(5 points)** Change this to use a different, more complex dataset.

**(5 points)** Recall that the Random Forest is an ensemble-based approach 
uses multiple decision trees. Replace the Decision Tree with the Random Forest Classifier. Run it on your dataset with 10, 25, and 50 estimators using both gini and entropy as separators. Create a table showing the results and add it to the PDF with your written answers.

**(5 points)** Doing hyperparameter search by hand is annoying. Sklearn makes it easy
for us with the GridSearchCV class. This lets us provide
a list of models and hyperparameters, and it does cross-validation for each 
model and parameter combination and summarizes the result. Very handy!

This example compares two different ensemble methods: Random Forest and Histogram Gradient Boosting. 
Modify it to:

a) test 5, 10, 15, and 20 estimators for Random Forest.
b) test 25, 50, 75, and 100 iterations for Histogram Boosting.
c) do 5 splits.

**(5 points)** The last part shows how to use plotly to generate a scatterplot
showing your results. Generate a plot and add it to the PDF with your answers.

#### Problem 2: Belief networks. 

For this problem, you'll be using the [pgmpy](https://pgmpy.org/) library for probabilistic inference. 
To begin, you'll want to install this.

To start, take a look at the alarm.py file, which encodes the earthquake example from class.
At the bottom is an example of how to query the network to find the probability that John calls given Earthquake. 

**(5 points)**  Add additional queries to determine:
- the probability of Mary Calling given that John called
- The probability of both John and Mary calling given Alarm
- the probability of Alarm, given that Mary called.
Include each of your queries in alarm.py

Add a main that executes each of your queries.

**(5 points)**  Next, consider the carnet.py file. This contains the Bayesian network representing the car starting problem.
To begin, ask the following queries:
- Given that the car will not move, what is the probability that the battery is not working?
- Given that the radio is not working, what is the probability that the car will not start?
- Given that the battery is working, does the probability of the radio working change if
we discover that the car has gas in it?
- Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?
- What is the probability that the car starts if the radio works and it has gas in it?
Include each of your queries in carnet.py. Also, please add a main
that executes your queries. 

**(5 points)**  3. Last, we will add an additional node to the network, called KeyPresent, that indicates whether or not we have the key for the car.
  This is a Categorical variable with two state values, yes and no. The prior for 'yes' is 0.7.
  
  KeyPresent should only affect Starts. Add an edge to starts and update the CPD to indicate that:
<pre>
P(starts | gas, ignition, keyPresent) = 0.99
P(starts | gas, !ignition, keyPresent) = 0.01
P(starts | !gas, ignition, keyPresent) = 0.01
P(starts | gas, ignition, !keyPresent) = 0.01
P(starts | !gas, !ignition, keyPresent) = 0.01
P(starts | !gas, ignition, !keyPresent) = 0.01
P(starts | gas, !ignition, !keyPresent) = 0.01 
P(starts | !gas, !ignition, !keyPresent) = 0.01
</pre>

Add a query showing the probability that the key is 
not present given that the car does not move.



#### Problem 3: Hidden Markov Models 

(Note: this is derived from an assignment in AAAI's Model Assignments workshop)

In this assignment you'll work with Hidden Markov models. We'll see how to use Monte Carlo simulation
to generate random sequences observations, how to figure out what state we are in given a set 
of observations, and the value of a sequence of hidden states. We'll apply this to robot localization
and part-of-speech tagging, as well as the all-important study of cat behavior. 


You'll be building off of the code presented in HMM.py. There's also some included data to use.
For each domain,  there's a .emit and a .trans file. .emit is our *emission* probabilities - how likely we are to see a particular variable, given a hidden state.
The .trans files are the *transition* probabilities; how likely we are to move between hidden states.

There are three domains: 
- cat: This is the example problem shown in class - what is my cat's mood?
- lander: The mars lander is floating above the terrain and trying to find a safe place to land.
- part of speech: These are English-language sentences; we will try to find the part of speech for each word.

The last pair of files is ambiguous_sents.obs and ambiguous_sents.tagged.obs. This is what you'll test your Viterbi part-of-speech implementation on.

Note that the .trans files use the '#' character to denote a starting state. You may choose whether
to implement forward and Viterbi using this or not.

**(5 points)**. Use the included code to implement load. Use cat as a sample file to work with. You should be able to do:

```
h = HMM.HMM()
h.load('cat')
h.transitions
{'#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'}, 'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'}, 'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'}, 'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}}
h.emissions
{'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'}, 'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'}, 'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}
```
You should store the transitions and emissions as dictionaries of dictionaries. 

Please add a unit test for load.


**(10 points)** One cool thing we can do with an HMM is Monte Carlo simulation. We'll do this
using generate. So implement that next. It should take an integer n, and return a Sequence of length n. To generate this, start in the initial state and repeatedly select successor states at random, using the transition 
probability as a weight, and then select an emission, using the emission probability as a weight. You may find either numpy.random.choice or random.choices very helpful here.
Be sure that you are using the transition probabilities to determine the next state, and not a uniform distribution!

Then, add a main that allows the user to generate a random sequence from the command line like so:

    python hmm.py cat --generate 20

or 

    python hmm.py partofspeech --generate 20

which generates 20 random observations.

Here are two sample observations:
```
grumpy grumpy happy happy happy hungry hungry grumpy hungry hungry grumpy grumpy grumpy happy happy happy happy hungry grumpy hungry
 meow meow purr meow meow silent meow silent purr silent silent meow silent silent meow purr meow meow silent silent
```
```
DET NOUN ADP NOUN CONJ DET VERB DET NOUN NOUN NOUN ADP DET NOUN
whose light for wall and the learned the hull postmaster trash in his peters
```

 In this step, you'll complete the emission and transition probabilities for the lander domain.

Our setup is this: Our lander is orbiting the surface of Mars,
and it needs to determine where it should land. Unfortunately, its sensors are not 
completely accurate! 

Please see the map in landermap.doc for a picture of the 
Martian landscape. The states are numbered 1,1 through 5,5.
Areas that are safe for us to land in are marked with an X.

We know the following things about Mars:
1. In general, we descend on a downward right diagonal, but with p=0.15 we 
are pushed directly right, and with p=0.15 we are pushed down. So from 1,1, we 
reach 2,2 with p=0.7, 1,2 with p=0.15, and 2,1 with p=0.15. 

2. Our sensors were damaged in the descent, and only get our location correct
with p=0.6. With p=0.1, they miscalculate our true location in each direction.
So, if we were truly in 2,2, our sensors would read 2,2 with p=0.6, 2,1 with p=0.1,
2,3 with p=0.1, 1,2 with p=0.1, and 3,2 with p=0.1. 

(We cannot go off the edges of the map - if we were to move
or sense locations off the edge, treat them as the border cell.)

**(5 points)** I've started filling in lander.emit and lander.trans. Complete these
models based on the description above.




**(10 points)** Next, you should implement the forward algorithm. This tells us, for a sequence of observations, the most likely
final state. 


You should be able to run this like so:

    python hmm.py cat --forward cat_sequence.obs

or 

    python hmm.py lander --forward lander_sequence.obs

Where do you get cat_sequence and lander_sequence from? From generate!
These are a sequence of emissions, such as "meow meow purr meow meow" or "1,1 2,2 2,2 3,3 4,3"
Take a look at ambiguous_sents.obs for a format.

Forward should predict the most probable state given the sequence of emisssions. For the lander, please
indicate whether it's safe to land or not.



**(15 points)** We can also use the HMM to infer sequences of hidden states. We'll use this
to determine parts of speech. The algorithm for doing this is Viterbi. This tells us, for a sequence of observations, the most likely sequence of hidden states. 
You should be able to run this like so:

    python hmm.py cat --viterbi cat_sequence.obs    

or

    python hmm.py partofspeech --viterbi ambiguous_sents.obs

You can compare your results to ambiguous_sents.tagged.obs.


Part 4. Utility.

This is a pen-and-paper (or typed) exercise. Please include a PDF with your answers
in your repository. 

Our Mars rover has been out collecting samples, when it detects that a sandstorm 
is coming. It needs to return to the safety of the charging station as quickly as possible. 

It knows that over rocky terrain it can go 2 km/h. Over sandy terrain it can go 3 km/h,
and over smooth terrain it can go 5 km/h. 

There are three routes it might choose from. Unfortunately, our terrain data for the three routes is incomplete,
so we only have estimates.

Route 1 is 5 km long. There is a 20% chance it is sandy, 30% chance it is smooth, and a 50% chance it is rocky.

Route 2 is 7 km long. There is a 40% chance it is sandy, a 20% chance it is smooth, and a 40 % chance it is rocky. 

Route 3 is 6 km long. There is a 50% chance it is sandy, a 40% chance it is smooth, and a 10% chance it is rocky.

**(5 points)** Which route should we pick? Show your work. You may find it easier
to convert km/hr to mins/km (how many minutes does it take to go 1km on each surface?)


Route 1 contains a crater. If the wall of the crater is intact, we can take a shortcut through the crater, which will save 20 minutes. If the wall has been damaged, we will need to go around, which will add 15 minutes to our journey. 
There is a 30% chance that the wall is damaged.

Route 3 contains a bridge. If that bridge is damaged, we will need to repair it, which will add 40 minutes to our time. There is a 60% chance that the bridge is damaged.

**(5 points)** Update your estimates for the travel time for each route. Now which route seems best?

Now we have an additional piece of potential information. There is an orbiting satellite that 
can tell us whether route 2 is rocky or not. If not, that would be great news, and would make it much more appealing!
The only problem is that the satellite is not yet in position. How long should we wait for the satellite?

**(2 points)** First: If the satellite said that route 2 was not rocky, how long would we expect it to take?

**(3 points)** Second: What's the probability that the satellite will tell us this?

**(2 points)** Third: If the satellite tells us route 2 is in fact rocky, what do we do? How long will that take?

**(3 points)** Last: given all of this, how long should we wait for the satellite? 
