# whoamI
Building AI course project : Whoami model with TF agents

## Summary

Here we propose a model of TF-agents to play the famous game "Who am I?".
Our version goes beyond the "basic" game because we will use other capabilities provided by AI (training of concurrent dynamic agents, image recognition and language models), to be able to apply this type of game to: a configurable number of characters (larger or smaller than the base game) and any photo of characters in a scene (so we can play with a family photo for example if we want).

Only a summarize of activities is provided here. Works are developed in the Colab file: [Colab Whoami model with TF agents](https://github.com/malter134/whoamI/blob/main/Whomai_model_with_TF_agents.ipynb)

## Background

There are already mathematical theories on gambling. These generally offer an approach in the form of probabilistic calculations on the chances of winning, highlighting the combinatorics of questions. Some of them have been recalled here in the sources.

Our version goes beyond the "basic" game because we will use other capabilities provided by AI (training of concurrent dynamic agents, image recognition and language models), to be able to apply this type of game to:
* a configurable number of characters (larger or smaller than the base game)
* any photo of characters in a scene (so we can play with a family photo for example if we want)

Apart from the "practical" side of this game, the interest of this game is to contribute at a theory of eliminatory questions around an "optimum" non-combinatorial question sequence to determine an individual in a group as efficiently as possible (interrogation strategy to find missing persons more quickly, etc.)

## How is it used?

In the first part, we develop game data (dataset) for the model:
  * Define the list of game questions with NLP traduction
  * Establish the dictionnary of people which will be used by the game environment with easyocr, huggingface, insighface, transformers and pipeline
  * Test these elements on several examples.

Example code to append person in the game data:
```
# The image of 'whoami standard game'...
image_src = load_image("http://lecoindespetits.l.e.pic.centerblog.net/o/160fdab2.png")
plt.title('Original Image')
plt.imshow(image_src)
plt.axis('off')
plt.show()

# Image with bad quality
# Try to use a small gaussian filter to have a better result
from PIL import Image
from scipy import ndimage
image_src = Image.fromarray(ndimage.gaussian_filter(np.array(image_src), sigma=1.25))

print("Compute the dictionnary of person...")
reset_dict_person()
add_dict_person(image_src, ocr=True)
print("Done.")
display_board()
```
<img src="https://github.com/malter134/whoamI/blob/main/capture-01.png" width="300">

In the second part, we develop the game model as a concurrent DQN agents working on a game environment:
   * Define the game environment with constraints and using the game data
   * Reuse the multi-agents interface
   * Instantiate the concurrent DQN agents (one by player) with a QNetwork
   * Train the model
   * Analyse the question sequences
   * Evaluate the model
   * Try to adapt the game at another images or questions with cosine similarity with sentence transformers and google image feature extraction

Game environment execution test:
```
environment = WhoAmIMultiAgentEnv()
utils.validate_py_environment(environment, episodes=20)
```

<img src="https://github.com/malter134/whoamI/blob/main/capture-02.png" width="500">


Example code part usage of agents:
```
import random

whoami_env = WhoAmIMultiAgentEnv()

ts = whoami_env.reset()
print('Reward:', ts.reward, 'Board:')
print_whoami(ts.observation)

random.seed(1)

# Players begin to choose a random person
selected_item = np.array([ 0, 0 ])
selected_item[0] = random.choice(range(nb_person))
selected_item[1] = selected_item[0]
while(selected_item[0] == selected_item[1]):
  selected_item[1] = random.choice(range(nb_person))

# Player 1 begin
player = 1

log=True

# test policy
# if test policy is 1, the questions will be computed by the environment
# also, the question will be random injected
# You can update this value to test other policy
test_policy =0
#test_policy = 1

# Copy of question list
r_question_list_for_player_1 = list(question_list_for_player_1)
r_question_list_for_player_2 = list(question_list_for_player_2)

# The question will be mean computed by environment
if test_policy == 1:
    # The question will be mean computed by environment
    # First question
    q_gen_1 = question_list.index(random.choice(question_list_for_player_1))
    q_gen_2 = question_list.index(random.choice(question_list_for_player_2))
    action = {
      'selected_item': selected_item,
      'question': np.array([ q_gen_1, q_gen_2 ]),
      'answer': np.array([ 0, 0 ]), # Always redefined by environment
      'player': player
    }

while not ts.is_last():
    if test_policy != 1:
        # Here the questions are random injected
        q_gen_1 = question_list.index(random.choice(r_question_list_for_player_1))
        q_gen_2 = question_list.index(random.choice(r_question_list_for_player_2))
        # To use the question only once
        r_question_list_for_player_1.remove(question_list[q_gen_1])
        r_question_list_for_player_2.remove(question_list[q_gen_2])
        action = {
          'selected_item': selected_item,
          'question': np.array([ q_gen_1, q_gen_2 ]),
          'answer': np.array([ 0, 0 ]), # Always redefined by environment
          'player': player
        }
    else:
        # Update only the player
        action['player'] = player

    other_player = 1 + player % 2
    print('Player:', player, '\n',
          'Question of player:', question_list[action['question'][player-1]], '\n',
          'Answer of other player:', yesno(action['answer'][other_player-1]), '\n',
          'Question of other player:', question_list[action['question'][other_player-1]], '\n',
          'Answer of player:', yesno(action['answer'][player-1]), '\n',
          'Reward:', ts.reward, 'Board:')
    ts = whoami_env.step(action)
    print_whoami(ts.observation)
    print_seq_questions(whoami_env.get_seq_questions())
    player = other_player
```

Topics have been developed in a Colab project.

You shall update the "basic" Colab configuration on each part as described in the Colab project.

## Data sources and AI methods
Game datas are computed in the topics in using some samples of images and a list of questions.
AI methods are described in the array below.

| AI methods                                                                                      | Description                                 |
| ----------------------------------------------------------------------------------------------  | ------------------------------------------- |
| Transformers and pipelines<br/>(NLP traduction, easyocr, huggingface, insighface)               | Build game datas:<br/>- Questions are translated\n- Person images are bounded and split with face and person recognition<br/>- For each person identified, get an answer on all the available questions<br/>- If the name of person exist in the image, take it |
| DQN TF-agents (QNetwork)                                                                        | Build game model with DQN concurrent TF-agents:<br/>- Define an environment game<br/>- Define a TF-agents (one per player) to learn to play |
| Sentence/image cosine similarity<br/>(sentence transformers and google image feature extraction pipeline)| Use another images or questions that game defined | 
## Challenges

Topic doesn't develop a model on the "lie factor" to better account for the fact that the adversary may lie to a question.

To be applicable, the volume of data shall be much larger and the model shall be trained extensively on this volume of data (images, questions). We have defined here only an example.

The study of question sequences can be developed on several other themes: Which type of questions are better? Could we define weights for the questions?...    

Anonymous of recorded data used (images) to train the model shall be taken in consideration to deploy a solution with a model computed to recognize an person in a group with a sequence of questions.   

## What next?

This project could be grow in the development of application on the security domain.

## Acknowledgments

* HOW TO WIN THE GAME “WHO IS IT?”: THANKS TO MATHEMATICS, Pierre-Luc Racine, March 16, 2020, https://urbania.fr/article/comment-gagner-au-jeu-qui-est-ce-grace-aux-mathematiques
* Optimal Strategy in “Guess Who?”: Beyond Binary Search, Mihai Nica, January 19, 2016, https://arxiv.org/pdf/1509.03327
* Multi-Agent Reinforcement Learning with TF-Agents, Dylan Cope, Jun 2020, https://github.com/DylanCope/Multi-Agent-RL-with-TF/blob/master/DMARL%20with%20TF-Agents.ipynb
* Training a Deep Q Network with TF Agents, Tensor Flow, https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=en
