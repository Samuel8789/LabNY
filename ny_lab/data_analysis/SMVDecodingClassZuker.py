# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:28:59 2022

@author: sp3660
"""

#@markdown #### Overfitting
#@markdown To understand why cross validation is important, let's put ourselves in a frequent case: we have recorded a lot of neurons, but we have a limited amount of trials

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
np.random.seed(0)

# variables
n_neurons =  300 #@param {type:"number"}
n_trials =  80 #@param {type:"number"}

mpl.rcParams['figure.dpi'] = 150

#@markdown In this hypothetical scenario, the trials are labeled with respect to two experimental conditions, called ```A``` and ```B```.

#@markdown However, we will generate some neural activity that contains <b>no information</b> about the two conditions:
#@markdown we will literally use ```np.random.rand``` to generate it.

# data generation
coding_level =  0.1

activity = np.random.rand(n_trials, n_neurons)
activity = (activity < coding_level).astype(float) # some sparsity

# labels, half A and half B
labels = np.hstack([np.repeat(-1, n_trials/2), np.repeat(1, n_trials/2)])

# visualization
def visualize_AB_activity(labels, activity, labA='A', labB='B'):
  n_trials, n_neurons = activity.shape
  f, ax = plt.subplots(figsize=(10, 5))
  ax.set_xlabel('Trial')
  ax.set_ylabel('Neuron #')
  sns.despine(ax=ax)
  ax.text(n_trials/2, n_neurons*1.1, 'Trial label', ha='center', fontsize=8)
  for t in range(n_trials):
    if labels[t] == -1:
      ax.text(t, n_neurons*1.05, labA, color='r', fontsize=6)
    if labels[t] == 1:
      ax.text(t, n_neurons*1.05, labB, color='b', fontsize=6)
  for i in range(int(n_neurons)):
    nanact = np.copy(activity[:, i])
    nanact[nanact==0] = np.nan
    ax.plot(i+nanact, marker='|', linestyle='', color='k', alpha=0.5, markersize=3)
  return f, ax


f, ax = visualize_AB_activity(labels, activity)

#@markdown We will now try to decode A vs. B from this random gibberish.

#@markdown Let's choose a very simple model: ```sklearn.svm.LinearSVC``` implements a linear classifier using support vectors. I highly recommend to chech this [interactive SVM tutorial](https://lisyarus.github.io/webgl/classification.html) out to get a visual intuition on how SVMs work.<br>

from sklearn.svm import LinearSVC
mysvc = LinearSVC()

#@markdown Let's train our classifier on the generated data
_ = mysvc.fit(activity, labels)

#@markdown And test its performance to assess how well we can separate the two classes!
decoding_performance =  mysvc.score(activity, labels)

# visualize the performance
f, ax = plt.subplots(figsize=(3, 3))
sns.despine(ax=ax)
ax.axhline([0.5], linestyle='--', color='k')
ax.set_xticks([0])
ax.set_ylabel('Decoding Performance')
ax.set_ylim([0.3, 1.05])
ax.axhline([1.0], color='k', alpha=0.3, linewidth=0.5)
ax.set_xticklabels(['A vs. B'])

print("Performance:", decoding_performance)
ax.plot(decoding_performance, marker='d', markersize=10, alpha=0.7, label='A vs. B')
plt.legend()


#@markdown This is why we need to cross-validate our analysis!  We will do so by following a simple prescription:

#@markdown - separate training and testing trials
#@markdown - train our machinery on the _training_ trials (intuitively)
#@markdown - test it on the (you guessed it) _testing_ trials

#@markdown We will do it on many random divisions of training and testing trials, so to get a distribution of performance values.
from tqdm import tqdm

training_fraction =  0.8 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 100 #@param {type:"number"}

def cross_validated_decoding(labels, 
                             activity, 
                             training_fraction, 
                             cross_validations,
                             verbose=False):
  
  # selecting A and B trials
  A_trials = np.where(labels==-1)[0]
  B_trials = np.where(labels==1)[0]
  
  if verbose:
    print('A trials:\n', A_trials)
    print('B trials:\n', B_trials)
  
  len_A = len(A_trials)
  len_B = len(B_trials)

  performances = []
  
  # looping on the cross_validation folds
  if verbose:
    print("\nLooping on %u cross validation folds:" % cross_validations)
  for k in tqdm(range(cross_validations)):
    np.random.shuffle(A_trials)
    np.random.shuffle(B_trials)

    # selecting training and testing trials for condition = A
    training_A = A_trials[:int(training_fraction * len_A)]
    testing_A = A_trials[int(training_fraction * len_A):]
    
    # selecting training and testing trials for condition = B
    training_B = B_trials[:int(training_fraction * len_B)]
    testing_B = B_trials[int(training_fraction * len_B):]

    training_trials =  np.hstack([training_A, training_B])
    testing_trials = np.hstack([testing_A, testing_B])
    
    # train on the training trials
    mysvc.fit(activity[training_trials], labels[training_trials])
    
    # test on the testing trials
    score = mysvc.score(activity[testing_trials], labels[testing_trials])
    performances.append(score)
    
    if verbose:
      print("\nIteration %u" % k)
      print("Training on trials:\n", training_trials)
      print("Testing on trials:\n", testing_trials)
      print("Performance: %.2f\n" % score)

  return performances

performances = cross_validated_decoding(labels, 
                                        activity, 
                                        training_fraction, 
                                        cross_validations=5, 
                                        verbose=True)


performances = cross_validated_decoding(labels, 
                                        activity, 
                                        training_fraction, 
                                        cross_validations, 
                                        verbose=False)

# visualization
def visualize_decoding_results(performances, labA='A', labB='B'):
  f, ax = plt.subplots(figsize=(3, 3))
  sns.despine(ax=ax)
  ax.axhline([0.5], linestyle='--', color='k')
  ax.set_xticks([0])
  ax.set_ylabel('Decoding Performance')
  ax.set_ylim([0.3, 1.05])
  ax.axhline([1.0], color='k', alpha=0.3, linewidth=0.5)
  ax.set_xticklabels(['%s vs. %s' % (labA, labB)], rotation=30)

  print("Mean performance: %.3f" % np.mean(performances))
  ax.errorbar([0], np.mean(performances), np.std(performances), marker='d', capsize=5, markersize=10, alpha=0.7, label='%s vs. %s' % (labA, labB))
  plt.legend()
  return f, ax


f, ax = visualize_decoding_results(performances)

# visualization
def visualize_decoding_results(performances, labA='A', labB='B'):
  f, ax = plt.subplots(figsize=(3, 3))
  sns.despine(ax=ax)
  ax.axhline([0.5], linestyle='--', color='k')
  ax.set_xticks([0])
  ax.set_ylabel('Decoding Performance')
  ax.set_ylim([0.3, 1.05])
  ax.axhline([1.0], color='k', alpha=0.3, linewidth=0.5)
  ax.set_xticklabels(['%s vs. %s' % (labA, labB)], rotation=30)

  print("Mean performance: %.3f" % np.mean(performances))
  ax.errorbar([0], np.mean(performances), np.std(performances), marker='d', capsize=5, markersize=10, alpha=0.7, label='%s vs. %s' % (labA, labB))
  plt.legend()
  return f, ax


f, ax = visualize_decoding_results(performances)

#@markdown We can re-use our ```cross_validated_deocding()``` function to analyze the decoding performance of A vs. B from the (random) neural activity. Given the randomness, we expect a chance levels performance around 0.5.
A_fraction =  0.85 #@param {type:"slider", min:0.5, max:0.95, step:0.01}
labels = np.hstack([-1*np.ones(int(n_trials * A_fraction)), np.ones(n_trials - int(n_trials*A_fraction))])

training_fraction =  0.85 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 100 #@param {type:"number"}

performances = cross_validated_decoding(labels, activity,
                                        training_fraction,
                                        cross_validations)

visualize_decoding_results(performances)

#@markdown As done for the first part, we will generate synthetic neural activity that contains <b>no information</b> about the two conditions A and B.

# variables
n_neurons =  300 #@param {type:"number"}
n_trials =  100 #@param {type:"number"}
coding_level =  0.1

# random data generation
activity = (np.random.rand(n_trials, n_neurons) < coding_level).astype(float) # some sparsity

#@markdown However, class A will be much more represented than B
A_fraction =  0.85 #@param {type:"slider", min:0.5, max:0.95, step:0.01}
labels = np.hstack([-1*np.ones(int(n_trials * A_fraction)), np.ones(n_trials - int(n_trials*A_fraction))])

# visualization
f, ax = visualize_AB_activity(labels, activity)

#@markdown We can re-use our ```cross_validated_deocding()``` function to analyze the decoding performance of A vs. B from the (random) neural activity. Given the randomness, we expect a chance levels performance around 0.5.
A_fraction =  0.94 #@param {type:"slider", min:0.5, max:0.95, step:0.01}
labels = np.hstack([-1*np.ones(int(n_trials * A_fraction)), np.ones(n_trials - int(n_trials*A_fraction))])

training_fraction =  0.85 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 100 #@param {type:"number"}

performances = cross_validated_decoding(labels, activity,
                                        training_fraction,
                                        cross_validations)

visualize_decoding_results(performances)

#@markdown By playing with the ```A_fraction``` parameter above, you will notice that there is a clear correlation between the over-representation of a class and the false positive performance of our decoding analysis.
#@markdown <br><br> Let's analyze it systematically in a non-overfitting setup (n_trials > n_features).

n_neurons =  100 #@param {type:"number"}
n_trials =  1000 #@param {type:"number"}
cross_validations =  25 #@param {type:"number"}

f, ax = plt.subplots(figsize=(5, 4))
sns.despine(ax=ax)
ax.set_xlabel('Class A imbalance')
ax.set_ylabel('A vs. B Decoding Performance')

for A_fraction in np.linspace(0.5, 0.95, 9):
  activity = (np.random.rand(n_trials, n_neurons) < coding_level).astype(float) # some sparsity
  labels = np.hstack([-1*np.ones(int(n_trials * A_fraction)), np.ones(n_trials - int(n_trials*A_fraction))])
  performances = cross_validated_decoding(labels, 
                                          activity, 
                                          training_fraction, 
                                          cross_validations)
  ax.errorbar(A_fraction, np.mean(performances), np.std(performances), capsize=5, color='k', marker='d')
  
  labels = mysvc.predict(activity)

f, ax = plt.subplots(figsize=(3, 3))
sns.despine(ax=ax)
ax.hist(labels, bins=[-1.5,-0.5,0.5,1.5])
ax.set_xticks([-1, 1])
ax.set_xticklabels(['A', 'B'])
ax.set_xlim([-2, 2])
ax.set_xlabel('Predicted class')
_ = ax.set_ylabel('Count')

  
#@markdown How can we avoid these false positives? There are two main possibilities <br>(1) change the weight of both classes in the SVC algorithm using the keyword ```class_weight=balanced```<br>(2) resample (up, down, random) the training data in a balanced way
#@markdown <br><br> Let's try the second approach: we re-write our cross validation decoding function by adding a parameter ```n_resampling``` that specifies the number of random samples we will draw for the training set of each class. This will allow the decoder to be trained on the same number of trials for the class A and B.

n_resampling = 100 #@param {type:"number"}

A_fraction = 0.85 #@param {type:"slider", min:0.1, max:0.98, step:0.01}

activity = (np.random.rand(n_trials, n_neurons) < coding_level).astype(float) # some sparsity
labels = np.hstack([-1*np.ones(int(n_trials * A_fraction)), np.ones(n_trials - int(n_trials*A_fraction))])

#@markdown <br><br> <b>very important</b>: trial resampling must be performed <b>after</b> the training-testing division. Otherwise, we will end up with the same data point in training and testing data!

def balanced_cross_validated_decoding(labels, 
                                      activity, 
                                      training_fraction, 
                                      cross_validations,
                                      n_resampling, # added this keyword
                                      verbose=False):
  A_trials = np.where(labels==-1)[0]
  B_trials = np.where(labels==1)[0]
  if verbose:
    print('A trials:\n', A_trials)
    print('B trials:\n', B_trials)
  len_A = len(A_trials)
  len_B = len(B_trials)

  performances = []
  
  if verbose:
    print("\nLooping on %u cross validation folds:" % cross_validations)
  
  for k in range(cross_validations):
    np.random.shuffle(A_trials)
    np.random.shuffle(B_trials)

    # selecting training and testing trials for condition = A
    training_A = A_trials[:int(training_fraction * len_A)]
    testing_A = A_trials[int(training_fraction * len_A):]
    
    # selecting training and testing trials for condition = B
    training_B = B_trials[:int(training_fraction * len_B)]
    testing_B = B_trials[int(training_fraction * len_B):]

    # ===> resampling trials <=== 
    training_A = np.random.choice(training_A, n_resampling)
    training_B = np.random.choice(training_B, n_resampling)
    testing_A = np.random.choice(testing_A, n_resampling)
    testing_B = np.random.choice(testing_B, n_resampling)
    # =========================== 

    training_trials =  np.hstack([training_A, training_B])
    testing_trials = np.hstack([testing_A, testing_B])


    mysvc.fit(activity[training_trials], labels[training_trials])
    score = mysvc.score(activity[testing_trials], labels[testing_trials])
    performances.append(score)
    
    if verbose:
      print("\nIteration %u" % k)
      print("Training on trials:\n", training_trials)
      print("Testing on trials:\n", testing_trials)
      print("Performance: %.2f\n" % score)

  return performances


#@markdown The classifier will still have a small imbalance in favour of A, especially with extreme values of ```A_fraction```
performances = balanced_cross_validated_decoding(
    labels, 
    activity, 
    training_fraction=0.8, 
    cross_validations=100,
    n_resampling=n_resampling
    )

labels = mysvc.predict(activity)

f, ax = plt.subplots(figsize=(3, 3))
sns.despine(ax=ax)
ax.hist(labels, bins=[-1.5,-0.5,0.5,1.5])
ax.set_xticks([-1, 1])
ax.set_xticklabels(['A', 'B'])
ax.set_xlim([-2, 2])
ax.set_xlabel('Predicted class')
_ = ax.set_ylabel('Count')

visualize_decoding_results(performances)

#@markdown <br><br>Let's create again a session of random neural activity. This time we will add a slow-decaying convolution artifact that spans groups of ```trials_per_block``` trials. 
#@markdown <br><br>This will mimic a recording session made of blocks of ```trials_per_block``` trials each, separated from each other by enough time to avoid that the slow convolution effect to leak between blocks.<br>
from math import ceil
np.random.seed(0)

n_neurons = 200 #@param {type:"number"}
n_trials = 120 #@param {type:"number"}
trials_per_block =  8#@param {type:"number"}

#@markdown To make our life easier (not), we will again use unbalanced data
A_fraction = 0.7 #@param {type:"slider", min:0.1, max:0.98, step:0.01}

#@markdown A $\tau$ parameter will control the typical time scale of autocorrelation
tau = 3 #@param {type:"slider", min:1, max:10, step:0.01}

# generate activity
activity = (np.random.rand(n_trials, n_neurons) < coding_level).astype(float) # some sparsity
labels = np.hstack([-1*np.ones(int(n_trials * A_fraction)), np.ones(n_trials - int(n_trials*A_fraction))])

# generate the block structure
blocks_A = np.repeat(range(ceil(n_trials*A_fraction/trials_per_block)), trials_per_block)[:int(n_trials*A_fraction)]
blocks_B = np.repeat(range(ceil(n_trials*(1-A_fraction)/trials_per_block)), trials_per_block)[:(n_trials-int(n_trials*A_fraction))]
blocks_B += np.max(blocks_A)+1
blocks = np.hstack([blocks_A, blocks_B])

# add slow convolution within blocks

for block in np.unique(blocks):
  block_trials = np.where(blocks==block)[0]
  block_activity = activity[block_trials]
  T = len(block_trials)
  for t in range(1, T):
    M = np.eye(n_neurons)
    for i in range(n_neurons):
      if np.random.rand() < 1./tau:
        np.random.shuffle(M[i])
    block_activity[t] = np.dot(M, block_activity[t-1])
  activity[block_trials] = block_activity

# visualize activity and blocks
f, ax = visualize_AB_activity(labels, activity)

for block in np.unique(blocks):
  block_trials = np.where(blocks==block)[0]
  ax.fill_between(block_trials, n_neurons, alpha=0.3)
  
  #@markdown Now let's try our **best** cross-validated balanced analysis on this data!

training_fraction =  0.85 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 100 #@param {type:"number"}
n_resampling = 150 #@param {type:"number"}

performances = balanced_cross_validated_decoding(
    labels, 
    activity, 
    training_fraction=training_fraction, 
    cross_validations=cross_validations,
    n_resampling=n_resampling
    )

visualize_decoding_results(performances)


def block_structured_balanced_cross_validated_decoding(
    labels, 
    activity, 
    training_fraction, 
    cross_validations,
    n_resampling,
    block_index,
    verbose=False):
  
  A_trials = np.where(labels==-1)[0]
  A_blocks = np.unique(block_index[A_trials])

  B_trials = np.where(labels==1)[0]
  B_blocks = np.unique(block_index[B_trials])

  if verbose:
    print('A trials:\n', A_trials)
    print('B trials:\n', B_trials)
    print('A blocks:\n', A_blocks)
    print('B blocks:\n', B_blocks)
  
  len_A = len(A_blocks)
  len_B = len(B_blocks)

  performances = []
  
  if verbose:
    print("\nLooping on %u cross validation folds:" % cross_validations)
  
  for k in tqdm(range(cross_validations)):
    np.random.shuffle(A_blocks)
    np.random.shuffle(B_blocks)

    # selecting training and testing blocks for condition = A
    training_blocks_A = A_blocks[:int(training_fraction * len_A)]
    testing_blocks_A = A_blocks[int(training_fraction * len_A):]

    # defining training and testing trials fromt the selected blocks
    training_trials_A = np.hstack([np.where(block_index == b)[0] for b in training_blocks_A])
    testing_trials_A = np.hstack([np.where(block_index == b)[0] for b in testing_blocks_A])

    # selecting training and testing blocks for condition = B
    training_blocks_B = B_blocks[:int(training_fraction * len_B)]
    testing_blocks_B = B_blocks[int(training_fraction * len_B):]

    # defining training and testing trials fromt the selected blocks
    training_trials_B = np.hstack([np.where(block_index == b)[0] for b in training_blocks_B])
    testing_trials_B = np.hstack([np.where(block_index == b)[0] for b in testing_blocks_B])

    # ===> resampling trials <=== 
    training_trials_A = np.random.choice(training_trials_A, n_resampling)
    training_trials_B = np.random.choice(training_trials_B, n_resampling)
    testing_trials_A = np.random.choice(testing_trials_A, n_resampling)
    testing_trials_B = np.random.choice(testing_trials_B, n_resampling)
    # =========================== 

    # concatenating data from sampled blocks
    training_trials =  np.hstack([training_trials_A, training_trials_B])
    testing_trials = np.hstack([testing_trials_A, testing_trials_B])

    # training the classifier
    mysvc.fit(activity[training_trials], labels[training_trials])
    
    # testing the classifier
    score = mysvc.score(activity[testing_trials], labels[testing_trials])
    performances.append(score)
    
    if verbose:
      print("\nIteration %u" % k)
      print("Training on trials:\n", training_trials)
      print("Testing on trials:\n", testing_trials)
      print("Performance: %.2f\n" % score)

  return performances

#@markdown Let's see how it works on this data!
training_fraction =  0.8 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 100 #@param {type:"number"}
n_resampling = 150 #@param {type:"number"}

performances = block_structured_balanced_cross_validated_decoding(
    labels, 
    activity, 
    training_fraction=training_fraction, 
    cross_validations=cross_validations,
    n_resampling=n_resampling,
    block_index=blocks, # <==== the block index!
    verbose=False
    )

visualize_decoding_results(performances)

#@markdown Let's create an example of activity that actually responds to a variable.

#@markdown Let's imagine we record from the visual cortex of a subject exposed to two different visuali stimuli, A and B.

#@markdown We create a synthetic version of this scenario by sampling from two gaussian distributions, one for A and one for B, with unitary diagonal covariance and random centers in the activity space.

#@markdown The two centers are placed at a mutual distance that is controlled by the ```AB_separation``` parameter. After sampling the data, we will binarize them with a thresholding non linearity.
np.random.seed(0)

AB_separation = 1.4 #@param {type:"slider", min:0.1, max:5, step:0.1}

n_neurons = 100 #@param {type:"number"}
n_trials = 120 #@param {type:"number"}

def generate_AB_activity(n_neurons, n_trials, separation):
  coding_level = 0.25
  # generate activity for stimulus A
  mean_A = np.random.rand(n_neurons) * separation # define the mean
  cov_A = np.eye(n_neurons) # unitary diagonal covariance matrix

  activity_A = np.random.multivariate_normal(
      mean_A, cov_A, int(n_trials/2))  # just sample from the gaussian
  activity_A = (np.abs(activity_A) < coding_level).astype(float)

  # generate activity for stimulus B
  mean_B = np.random.rand(n_neurons) * separation
  cov_B = np.eye(n_neurons)

  activity_B = np.random.multivariate_normal(
      mean_B, cov_B, int(n_trials/2))
  activity_B = (np.abs(activity_B) < coding_level).astype(float)

  # put the activity together
  V1_activity = np.vstack([activity_A, activity_B])
  stimulus_labels = np.hstack([-1*np.ones(int(n_trials/2)), np.ones(int(n_trials/2))])

  # order the activity for plotting purposes
  selectivity = np.mean(activity_A, 0) - np.mean(activity_B, 0)
  order = np.argsort(selectivity)
  V1_activity = V1_activity[:, order]

  # return activity and labels
  return V1_activity, stimulus_labels


# let's call the function to get the activity
V1_activity, stimulus_labels = generate_AB_activity(n_neurons, 
                                                    n_trials, 
                                                    AB_separation)

# let's visualize it
f, ax = visualize_AB_activity(stimulus_labels, V1_activity)

#@markdown Looks like there is some visible (almost?) difference, but is it enough to decode it? Let's try with our cross validated balanced pipeline.

training_fraction =  0.8 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations =  20#@param {type:"number"}
n_resampling = 150 #@param {type:"number"}

performances = balanced_cross_validated_decoding(
    labels=stimulus_labels,
    activity=V1_activity,
    training_fraction=training_fraction,
    cross_validations=cross_validations,
    n_resampling=n_resampling
    )

f, ax = visualize_decoding_results(performances)

#@markdown We can't use a t-test against 0.5 because it depends on the number of cross validations, which is completely under our control!
# import scipy
# scipy.stats.ttest_1samp(performances, 0.5)

@markdown The mean performance is larger than 0.5, but how can we tell if it is significant?

#@markdown Let's follow the null model principle: we keep the data intact and we only break its relationship with the ```A``` and ```B``` labeling

#@markdown We will repeat the shuffling ```nshuffles``` times, and each time we will run the whole cross validated analysis on the resulting null data set.
nshuffles = 50 #@param {type:"number"}

null_performances = []

for i in tqdm(range(nshuffles)):
  null_labels = np.copy(stimulus_labels) # these will be shuffled
  np.random.shuffle(null_labels) # <- here

  null_p = balanced_cross_validated_decoding(
      labels=null_labels,
      activity=V1_activity,
      training_fraction=training_fraction,
      cross_validations=cross_validations,
      n_resampling=n_resampling
      )
  null_performances.append(np.mean(null_p))

#@markdown Let's visualize the performance against the null model distribution and see if the effect is significant!

perf = np.mean(performances)

def visualize_performance_nullmodel(perf, null_performances):
  import scipy
  # computing the P value of the z-score
  from scipy.stats import norm
  null_mean = np.nanmean(null_performances)
  z = (perf - null_mean) / np.nanstd(null_performances)
  p = norm.sf(abs(z))

  def p_to_ast(p):
      if p < 0.001:
          return '***'
      if p < 0.01:
          return '**'
      if p < 0.05:
          return '*'
      if p >= 0.05:
          return 'ns'

  # visualizing
  f, ax = plt.subplots(figsize=(6, 3))
  kde = scipy.stats.gaussian_kde(null_performances)
  null_x = np.linspace(0., 1.0, 100)
  null_y = kde(np.linspace(0.1, 0.9, 100))
  ax.plot(null_x, null_y, color='k', alpha=0.5)
  ax.fill_between(null_x, null_y, color='k', alpha=0.3)
  ax.text(null_x[np.argmax(null_y)], np.max(null_y)*1.05, 'null model', ha='right')
  sns.despine(ax=ax)
  ax.plot([perf, perf], [0, np.max(null_y)], color='red', linewidth=3)
  ax.text(perf, np.max(null_y)*1.05, 'data', ha='left', color='red')
  ax.set_xlabel('Decoding Performance (Stimulus)')
  ax.set_xlim([0.2, 0.8])
  ax.text(0.85, 0.05*np.max(null_y), '%s\nz=%.1f\nP=%.1e' % (p_to_ast(p), z, p), ha='center')
  ax.plot(null_performances, np.zeros(len(null_performances)), linestyle='', marker='|', color='k')
  _ = ax.plot([null_mean, null_mean], [0, kde(null_mean)], color='k', linestyle='--')
  return f, ax, z, p

f, ax, z, p = visualize_performance_nullmodel(perf, null_performances)

#@markdown Let's create some synthetic data that responds to the visual stimuli
np.random.seed(0)

AB_separation = 3 #@param {type:"slider", min:0.1, max:5, step:0.1}

n_neurons = 120 #@param {type:"number"}
n_trials = 180 #@param {type:"number"}
V1_activity, stimulus_labels = generate_AB_activity(n_neurons, 
                                                    n_trials, 
                                                    AB_separation)

f, ax = visualize_AB_activity(stimulus_labels, V1_activity)

ax.fill_between([0, n_trials/2], n_neurons+1, color='r', alpha=0.1)
ax.fill_between([n_trials/2+1, n_trials], n_neurons+1, color='b', alpha=0.1)

#@markdown We can easily (and legitimately!) decode A vs. B from the sampled activity:

training_fraction =  0.8 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 50 #@param {type:"number"}
n_resampling = 100 #@param {type:"number"}

performances = balanced_cross_validated_decoding(
    labels=stimulus_labels,
    activity=V1_activity,
    training_fraction=training_fraction,
    cross_validations=cross_validations,
    n_resampling=n_resampling
    )

f, ax = visualize_decoding_results(performances)

#@markdown As we mentioned above, the subject performs with ~80% accuracy the task. Let's create a simple implementation of this behavior

action_labels = np.copy(stimulus_labels)
action_labels[int(2*n_trials/5) : int(n_trials/2)] = 1
action_labels[int(9*n_trials/10):] = -1
f, ax = visualize_AB_activity(action_labels, V1_activity, 'L', 'R')

ax.fill_between([0, n_trials/2], n_neurons+1, color='r', alpha=0.1)
ax.fill_between([n_trials/2+1, n_trials], n_neurons+1, color='b', alpha=0.1)

#@markdown We know that our synthetic V1 activity does not respond to ```action```, as we sampled it according to the ```stimulus``` value.

#@markdown But what happens if we apply our best balanced cross validated pipeline to decode ```action``` from V1 activity?

training_fraction =  0.8 #@param {type:"slider", min:0.1, max:0.98, step:0.01}
cross_validations = 25 #@param {type:"number"}
n_resampling = 100 #@param {type:"number"}

performances = balanced_cross_validated_decoding(
    labels=action_labels, # <==== trials labeled according to action
    activity=V1_activity,
    training_fraction=training_fraction,
    cross_validations=cross_validations,
    n_resampling=n_resampling
    )

nshuffles = 25 #@param {type:"number"}

null_performances = []

for i in tqdm(range(nshuffles)):
  null_labels = np.copy(stimulus_labels)
  np.random.shuffle(null_labels)

  null_p = balanced_cross_validated_decoding(
      labels=null_labels,
      activity=V1_activity,
      training_fraction=training_fraction,
      cross_validations=cross_validations,
      n_resampling=n_resampling
      )
  null_performances.append(np.mean(null_p))

f, ax, z, p = visualize_performance_nullmodel(
    np.mean(performances), 
    null_performances)

