"""
This code is a computational model of a task called 'cue-target orientation'
For running the model, we used Spyder in python 3.9
For making the model work, we modified line 115 of the file 'pymdp.algos.mmp' , which now should look like this:  err = (coeff * (lnA*1.05) + lnB_past + lnB_future) - coeff * lnqs
"""

import numpy as np
import pymdp
from pymdp import utils, inference
from pymdp.agent import Agent
from pymdp.envs import Env
from pymdp.algos import run_mmp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random


"""
Output functions
This section is for creating a dataframe at the end and having the final loop more empty
"""

def output1(data_trial, timestep, qs, observation, F):
    # KL divergence for timesteps 1,2,3
    if timestep == 1:
        data_trial[6] = F[0]*-1
    elif timestep == 2:
        data_trial[7] = F[0]*-1
    elif timestep == 3:
        data_trial[8] = F[0]*-1

    # qs timesteps for timesteps 1,2,3
    if timestep == 0:
        data_trial[12] = qs[0][0][1][timestep]
    elif timestep == 1:
        data_trial[13] = qs[0][0][1][timestep]
    elif timestep == 2:
        data_trial[14] = qs[0][0][1][timestep]
    elif timestep == 3:
        data_trial[15] = qs[0][0][1][timestep]

    # put data in dataframe
    if observation[0] == 0:
        data_trial[2] = 45  # diamond
    elif observation[0] == 1:
        data_trial[2] = 0  # square
    elif observation[1] == 0:
        data_trial[3] = 45  # left
    elif observation[1] == 1:
        data_trial[3] = 315  # right

    #keypress (left or right)
    if timestep == 1:
        if qs[0][0][0][0] > 0.5:
            data_trial[1] = 'left'
        elif qs[0][0][0][0] < 0.5:
            data_trial[1] = 'right'
        else:
            if random.random() > 0.5:
                data_trial[1] = 'left'
            else:
                data_trial[1] = 'right'

    # QS that infer_states returns for timesteps 1,2,3
    if timestep == 1:
        if qs[0][0][0][0] > 0.5:
            data_trial[9] = qs[0][0][0][0]
        elif data_trial[1] == 'right':
            data_trial[9] = qs[0][0][0][1]
    elif timestep == 2:
        if qs[0][0][0][0] > 0.5:
            data_trial[10] = qs[0][0][0][0]
        else:
            data_trial[10] = qs[0][0][0][1]
    elif timestep == 3:
        if qs[0][0][0][0] > 0.5:
            data_trial[11] = qs[0][0][0][0]
        else:
            data_trial[11] = qs[0][0][0][1]


def output2(data_trial, env, trial): #second part of the output
    # calculate  KL divergence
    data_trial[16] = (data_trial[7]-data_trial[6]) #KLD of 2-1
    data_trial[17] = (data_trial[8]-data_trial[7]) #KLD of 3-2
    data_trial[18] = (data_trial[8]-data_trial[6]) #KLD of 3-1

    # trial number
    data_trial[0] = trial

    # frequency/trial type (80% or 20%)
    data_trial[4] = env.get_frequency()

    # A matrix over time
    data_trial[19] = my_agent.A[0][0][0][0] #left-diamond
    data_trial[20] = my_agent.A[0][1][1][0] #right-square

    # correct or incorrect
    if data_trial[1] == 'left' and data_trial[3] == 45 or data_trial[1] == 'right' and data_trial[3] == 315:
        data_trial[5] = 1
    else:
        data_trial[5] = 0


"""
Defining the relevant parameters about the the cue-target orientation experiment 
"""
cues_names = ['diamond', 'square', 'nothing']  # nothing means no cue
feedback_names = ['left', 'right', 'nothing']  # nothing means straight patch
states = ['left', 'right']  # hidden factor 1 with 2 hidden states
timesteps = ["1", "2", "3", "4"]  # hidden factor 2
factors = [states, timesteps]  # hidden factors

n_obs = [len(cues_names), len(feedback_names), len(timesteps)] #number of possible observations
n_states = len(factors)  # number of hidden factors

"""
List of trials specifying the cue and the hidden state 1, and randomize it
This list will be used for the model environment or generative process later in the code
 """
trials_hid_states1 = []

def listHS():

    for i in range(80):  # 80% of diamonds are left and squares right
        trials_hid_states1.append("diam_left")
        trials_hid_states1.append("squ_right")

    for i in range(20):
        trials_hid_states1.append("diam_right")
        trials_hid_states1.append("squ_left")
    random.shuffle(trials_hid_states1)

listHS()

"""
Defining te A matrix
A[0] represents the relation of cues with the hidden state left/right. If you plot it, you can see that the cues are only visible on A[0][0], that represents timestep 0.
A[1] represents the relation of the gabor patches with the hidden state left/right. Gabor patches are only visible in timesteps 2 and 3, so A[1][2] and A[1][3]
modifying the value agent_prob, you can change the knowledge of the agent
A[2] represents the timestep probability for each timestep
"""
# Creating A matrix
A = utils.obj_array(len(n_obs))  # matrix observations define how is A
A[0] = np.ndarray((3, 2, 4))
A[1] = np.ndarray((3, 2, 4))  # create array 3 rows, 2 columns and 4 matrices for A[0] and A[1]
A[0] = np.zeros((3, 2, 4))
A[1] = np.zeros((3, 2, 4))  # make A[0] and A[1] full of zeros
A[2] = np.ndarray((4, 2, 4)) #Create A[2]
A[2] = np.zeros((4, 2, 4))

# SET THE AGENT KNOWLEDGE. If you set the probability to 0.5, the agent starts knowing nothing about the distribution. 
#If you put 0.8, the agent will know the generative pocess of A[0], the real probabilities
agent_prob = 0.8 #The probability is set to 0.8 because the agent already learned the model. This is implemented for improving the model in the future

# Filling A matrix
for i in range(3):  # the first axis up/down
    if i == 2:
        A[0][i, :, 1:4] = 1  # timestep 1 to 3 they do not see any cue
        A[1][i, :, 0:2] = 1  # timestep 0 and 1 they do not see any feedback
    for j in range(2):  # the second axis of left/right
        for k in range(4):  # the third axis of timesteps
            if k == 0:
                if agent_prob == 0.5:
                    # for 0.5
                    # the first matrix shows the probability of association of each cue with the gabor patch, it starts 0.5 everywhere
                    A[0][0:2, j, k] = 0.5
                    # for 0.8 learned
                else:
                    A[0][0, 0, k] = 0.8
                    A[0][1, 1, k] = 0.8
                    A[0][0, 1, k] = 0.2
                    A[0][1, 0, k] = 0.2
            if k > 1:
                # in the two last timesteps, the correlation feedback/hidden state (left/right) is an identity matrix
                A[1][0:2, :, k] = np.eye(2)
#Defining A[2] for making the agent know that it is in a determined timestep
for i in range(len(states)):
    A[2][:, i, :] = np.eye(4)
    
"""
This part of the code is dedicated to print the matrices of the generative model
For printing a matrix in particular, set plot to 'on'
"""

# Plotting the matrices
# Parameters that we need for plotting
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sn.set(style='ticks', font_scale=1, rc={
    'axes.linewidth': 1,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor': 'Black',
    'xtick.color': 'Black',
    'ytick.color': 'Black', })
sn.plotting_context()

# Plot A[0]
# For plotting, set plot = on
plot = 'off'
if plot == 'on':
    fig = plt.figure(figsize=(10, 4))
    # loop over A[0] and make a subplot with heatmap
    for i, little_a in enumerate(range(4)):
        # make a 1x4 subplot, 1st subplot
        axa0 = fig.add_subplot(1, A[0].shape[2], i+1)
        sn.heatmap(A[0][:, :, i], ax=axa0, vmax=1)

# Plot A[1]
# For plotting, set plot = on
plot = 'off'
if plot == 'on':
    fig2 = plt.figure(figsize=(10, 4))
# loop over A[1] and make a subplot with heatmap
    for i, little_a in enumerate(range(4)):
        # make a 1x4 subplot, 1st subplot
        axa1 = fig2.add_subplot(1, A[1].shape[2], i+1)
        sn.heatmap(A[1][:, :, i], ax=axa1)

# Plot A[2]
# For plotting, set plot = on
plot = 'off'
if plot == 'on':
    fig2 = plt.figure(figsize=(10, 4))
# loop over A[2] and make a subplot with heatmap
    for i, little_a in enumerate(range(4)):
        # make a 1x4 subplot, 1st subplot
        axa1 = fig2.add_subplot(1, A[2].shape[2], i+1)
        sn.heatmap(A[2][:, :, i], ax=axa1)

"""
Defining the B matrix
B[0] represents the transition of the hidden state left/right. Within a trial, it is a 2x2 identity matrix, so it does not change
B[1] represents the transition between timesteps. It informs that after timestep 0 is timestep 1, etc. Is a 4x4 matrix
"""
B = utils.obj_array(n_states)  # matrix observations
B[0] = np.ndarray((2, 2, 1))
B[1] = np.ndarray((4, 4, 1))
B[0] = np.zeros((2, 2, 1))
B[1] = np.zeros((4, 4, 1))

B[0][:, :, 0] = np.eye(2)  # transition left/rigt
B[1][:, :, 0] = np.roll(np.eye(4), -1)  # transition between timesteps

# Plot B[0]
# For plotting, set plot = on
plot = 'off'
if plot == 'on':
    fig2 = plt.figure(figsize=(10, 4))
    axb0 = fig2.add_subplot(1, 1, 1)  # make a 1x4 subplot, 1st subplot
    sn.heatmap(B[0][:, :, 0], ax=axb0)

# Plot B[1]
# For plotting, set plot = on
plot = 'off'
if plot == 'on':
    fig2 = plt.figure(figsize=(10, 4))
    axb1 = fig2.add_subplot(1, 1, 1)  # make a 4x4 subplot, 1st subplot
    sn.heatmap(B[1][:, :, 0], ax=axb1)

"""
Defining D matrix
"""
num_states = [2, 4]  # 2 hid states 1 (L/R) and 4 timesteps
# this makes one .50/.50 vector, and one .25/.25/.25/.25 vector
D = pymdp.utils.obj_array_uniform(num_states)
# replace the 0.25 uniform one with a onehot
D[1] = utils.onehot(0, num_states[1])

"""
Defining a subclass "New agent" that permits us to modify some of the functions of the module
This method is called: subclassing and overriding
The current functions that are modified are: infer_states 
"""

class NewAgent(Agent):
    def __init__(self, A, B, C, D, inference_algo, use_BMA, policy_sep_prior, save_belief_hist, modalities_to_learn):
        """ Inherit all of the methods and attributes of `Agent` """
        super().__init__(A,
                         B,
                         C,
                         D,
                         inference_algo="MMP",
                         # only learn the first observation modality, corresponding to the Left-RIght Gabor stimulus contingencies
                         modalities_to_learn=[0],
                         # learning rate for updating posterior over observation likelihood (`eta` in SPM)
                         lr_pA=100.0,
                         use_BMA=False,
                         policy_sep_prior=True,
                         save_belief_hist=True,
                         pA=A
                         )

    def infer_states(self, observation, distr_obs=False):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.
        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """

        observation = tuple(observation) if not distr_obs else observation

        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo == "VANILLA":
            if self.action is not None:
                empirical_prior = control.get_expected_states(
                    self.qs, self.B, self.action.reshape(1, -1)  # type: ignore
                )[0]
            else:
                empirical_prior = self.D
            qs = inference.update_posterior_states(
                self.A,
                observation,
                empirical_prior,
                **self.inference_params
            )
        elif self.inference_algo == "MMP":
            self.prev_obs.append(observation)
            if len(self.prev_obs) > self.inference_horizon:
                latest_obs = self.prev_obs[-self.inference_horizon:]
                latest_actions = self.prev_actions[-(
                    self.inference_horizon-1):]
            else:
                latest_obs = self.prev_obs
                latest_actions = self.prev_actions
            qs, F = inference.update_posterior_states_full(
                self.A,
                self.B,
                latest_obs,
                self.policies,
                latest_actions,
                prior=self.latest_belief,
                policy_sep_prior=self.edge_handling_params['policy_sep_prior'],
                **self.inference_params
            )
            self.F = F  # variational free energy of each policy

        if hasattr(self, "qs_hist"):
            self.qs_hist.append(qs)
        self.qs = qs

        return qs, F

"""
Creating my_agent, from Newagent, using MMP as the inference algorithm
"""
my_agent = NewAgent(A=A, B=B, C=None, D=D, inference_algo="MMP", use_BMA=False,
                    policy_sep_prior=True, save_belief_hist=True, modalities_to_learn=[0])

"""
Defining the environment
The main parts of the environment are the hidden state and the observation of the agent or step function
"""


class custom_env(Env):
    def __init__(self):  # initiate self
    #This part creates a vector that tells the agent in which timestep they are in
        self.start = custom_env.one_hot(4, 0)
        self.state = self.start
        self.B = B
    @staticmethod
    def one_hot(n, idx):
        vec = np.zeros(n)
        vec[idx] = 1.0
        return vec

    # This function gets a hidden state from the list of hidden states defined earlier
    def get_hid_state_1(self):
        # Decide the hidden state randomly for each trial
        self.hid_state_raw = trials_hid_states1.pop(
            1)  # take a hidden state from the hs list
        if self.hid_state_raw == "diam_left" or self.hid_state_raw == "squ_left":
            self.hid_state1 = [0, 1]  # [0,1] is leftt
        else:
            self.hid_state1 = [1, 0]  # Right

    # This function translates the hidden state of the trial to numerical frequency
    def get_frequency(self):
        if self.hid_state_raw == "diam_left" or self.hid_state_raw == "squ_right":
            self.frequency = 80
        else:
            self.frequency = 20
        return self.frequency


    def step(self, timestep):  # obs are martix rows
        """
        Takes the timestep and the hidden state of the trial and returns the observations of the participant (cues and feedback)

        Parameters:
        Timestep: 0 to 4 within a trial
        Hidden state: Left([0,1]) or right([1,0])

        Returns: 
        Observation: [which cue is the participant seeing, which gabor patch is the participant seeing, which timestep are we in]
        """
        # The trial has 4 timesteps, depending in which one we are in,
        if timestep == 0:  # T - cue onset phase of the trial
            # get the hidden state 1 (left or right)
            if self.hid_state_raw == "diam_left" or self.hid_state_raw == "diam_right":
                obs_cue = 0  # diamond
            else:
                obs_cue = 1  # square

            obs_feedback = 2  # straight
            t = 0
        # what is this branch? explain in words related to the experiment and conditions
        elif timestep == 1:  # T+1 decision phase of the trial, the participants need to response with a left or right decision
            obs_cue = 2  # nothing
            obs_feedback = 2
            t = 1
        # T+2 feedback phase of the trial, the participants see a left or right oriented gabor patch.
        elif timestep == 2:
            obs_cue = 2
            obs_feedback = self.hid_state1[0]  # 0 is Left, 1 is Right
            t = 2
        elif timestep == 3:  # T+3 last time step here was implemented for updating the model?
            obs_cue = 2
            obs_feedback = self.hid_state1[0]
            t = 3
        observation = [obs_cue, obs_feedback, t]
        return observation

# In this part, we define the timesteps(4) and the trials of our experiment
env = custom_env()  # for calling the functions
T = 4  # number of timesteps
N = 50  # number of trials

# In this part we define the columns of the csv output of the model
data = {'trial_counter': [], 'keypress': [], 'cue_ori': [], 'target_ori': [], 'frquency': [], 'correct': [], 'FE1': [], 'FE2': [], 'FE3': [],
        'qs_1': [], 'qs_2': [], 'qs_3': [], 'qs_tim_0': [], 'qs_tim_1': [], 'qs_tim_2': [], 'qs_tim_3': [], 'KL_2-1': [], 'KL_3-2': [], 'KL_3-1': [], 'D_L': [], 'S_R': []}
output = pd.DataFrame(data)
data_trial = [0] * len(data)

def main_loop(T, N, env, my_agent):  # this function runs the model
    """
    MAIN LOOP: This loop makes the agent go through the experiment
    The main parts are: Infering states, choosing action
    """
    if len(trials_hid_states1) < 200:
        listHS()
    for trial in range(1, N):  # N is number of trials (in this case 200)
        env.get_hid_state_1()  # replace with env.get_hidden_state()
        rst = my_agent.reset()  # reset agent

        for timestep in range(T):  # real timesteps within a trial (4)
            # we run the function step in the environment to get an observation (tuple with 2 numbers)
            observation = env.step(timestep)
            # with the observation we infer the posterior
            qs, F = my_agent.infer_states(observation)
            q_pi, efe = my_agent.infer_policies()
            chosen_action_id = my_agent.sample_action()

            # OUTPUT part 1
            output1(data_trial, timestep, qs, observation, F)

        # OUTPUT part 2
        output2(data_trial, env, trial)
        output.loc[trial-1] = data_trial

# RUNNING THE MODEL
main_loop(T, N, env, my_agent) 
