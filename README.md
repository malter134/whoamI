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

Extract code to build the dictionnary of person:
```
# Dictionnary person generation
...
# constant in pixels to delimit a zone arround the face (adjust if need)
delta_face_arround_ctx_x = 20
delta_face_arround_ctx_y = 70

# Load the insightface model to detect person in a image
def load_model_detection():
    path = huggingface_hub.hf_hub_download("public-data/insightface", "models/scrfd_person_2.5g.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 8
    session = ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )
    model = insightface.model_zoo.retinaface.RetinaFace(model_file=path, session=session)
    return model

# Method to detect a person in image, return bounding box
def detect_person(
    img: np.ndarray, detector: insightface.model_zoo.retinaface.RetinaFace
) -> tuple[np.ndarray]:
    bboxes, kpss = detector.detect(img)
    bboxes = np.round(bboxes[:, :4]).astype(int)
    return bboxes

# Extract the images of person from the source image
def extract_sub_images(image: np.ndarray, bboxes: np.ndarray) -> list[np.ndarray]:
    res = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox
        res.append(image[y1:y2, x1:x2])
    return res

# Init part (models and pipelines)
detector = load_model_detection()
detector.prepare(-1, nms_thresh=0.5, input_size=(640, 640))
face_detector = FaceAnalysis()
# ctx_id = 0 (GPUid), det_size of image
face_detector.prepare(ctx_id=0, det_size=(640,640))
reader = easyocr.Reader(['en','en']) # this needs to run only once to load the model into memory
vqa_pipeline = pipeline("visual-question-answering", device="cuda")

# Extract text of image (ocr)
def text_extract(image: np.ndarray) -> str:
    result = reader.readtext(image, detail = 0)
    # Remove the element with double because the name shall be unique...
    result = [r for r in result if result.count(r) == 1]
    res = ' '.join(result)
    return res

# Extract sub-image of person from the source image
def detect_extract(image: np.ndarray) -> np.ndarray:
    # Try with a complete person
    bboxes = detect_person(image, detector)
    # If not found, try with the face of person
    if len(bboxes) == 0:
        bboxes = []
        faces = face_detector.get(image)
        if len(faces) > 0:
          for i in range(len(faces)):
            x1, y1, x2, y2 = np.round(faces[i]['bbox']).astype(int)
            # Append delta arround the face to append the context and name
            if x1 - delta_face_arround_ctx_x < 0:
                x1 = 0
            else:
                x1 = x1 - delta_face_arround_ctx_x
            if y1 - delta_face_arround_ctx_y < 0:
                y1 = 0
            else:
                y1 = y1 - delta_face_arround_ctx_y
            if x2 + delta_face_arround_ctx_x > image.shape[1]:
                x2 = image.shape[0]
            else:
                x2 = x2 + delta_face_arround_ctx_x
            if y2 + delta_face_arround_ctx_y > image.shape[0]:
                y2 = image.shape[1]
            else:
                y2 = y2 + delta_face_arround_ctx_y
            x1_m = min(x1, x2)
            y1_m = min(y1, y2)
            x2_m = max(x1, x2)
            y2_m = max(y1, y2)
            bboxes.append([x1_m, y1_m, x2_m, y2_m])
    res = extract_sub_images(image, bboxes)
    return res

# Query on a image (caption). Return yes or no.
def query(image: np.ndarray, question: str) -> str:
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image) # Convert the NumPy array to a PIL Image object
    res = vqa_pipeline({"image": image, "question": question}, top_k=1)
    return res[0]['answer']

# The "dictionnary" of person
dict_person = []

# To reset the dict person
def reset_dict_person():
  dict_person.clear()

# To update the dict person with all infos on the person
def update_dict_person(person_id: int, person_img: np.ndarray, question_list, ocr=False):
  person_name = ''
  if (ocr == True):
    # Search if the name of person exist in the image (optional)
    person_name = text_extract(person_img)
    # To have a short name we keep here only the first part of long name
    person_name = person_name.split()[0]
  if len(person_name) < 2:
    # Ignore the result and use a generic name
    person_name = "Person" + str(person_id)
  questions = {}
  for q in question_list:
      questions[q] = query(person_img, q)
  # Check if person_id already exists in the list of dictionaries
  person_exists = False
  for person in dict_person:
      if person.get('id') == person_id:  # Use get() to avoid KeyError if 'id' is missing
          person_exists = True
          person['name'] = person_name
          person['img'] = person_img
          person['questions'] = questions
          break
  # If person_id doesn't exist, add a new dictionary to the list
  if not person_exists:
      dict_person.append({ 'id': person_id, 'name': person_name, 'img': person_img, 'questions': questions })
...
# To get a answer yes or no on a question on a person
def get_answer_on_person(question: str, person_id: int):
    if question not in dict_person[person_id]['questions']:
        return "undef"
    return dict_person[person_id]['questions'][question]

# To get a answer on a question for a list of person
# Return can be "yes", "no" or "undef"
def get_answer_on_list_person(question: str, sub_list_person_id = range(len(dict_person))):
    # Check "yes" case for all the person on the sub list of person
    res = "yes"
    for person in dict_person:
      for i in sub_list_person_id:
        if person['id'] == i:
          if person['questions'][question] == "no":
            res = "no"
            break
    if res == "yes":
      return "yes"
    # Check "no" case for all the person on the sub list of person
    res = "no"
    for person in dict_person:
      for i in sub_list_person_id:
        if person['id'] == i:
          if person['questions'][question] == "yes":
            res = "yes"
            break
    if res == "no":
      return "no"
    return "undef"
...
# To add a dict person from a image source
def add_dict_person(image_src: np.ndarray, ocr=False):
  nb_person = get_nb_person()
  for img in detect_extract(np.asarray(image_src)):
    update_dict_person(nb_person, img, question_list, ocr)
    nb_person +=1
```

Example code to append a person in the game data:
```
# The image of 'whoami standard game'...
image_src = load_image("http://lecoindespetits.l.e.pic.centerblog.net/o/160fdab2.png")
plt.title('Original Image')
plt.imshow(image_src)
plt.axis('off')
plt.show()
...
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

Example code part to test the game environment:
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

Example code part use of agents with similarity:
```
from itertools import cycle

ts = tf_ttt_env.reset()
player_1.reset()
player_2.reset()
print('Start:')
# arbitrary starting point to add variety
random.seed(1)
# Selected item is an initial condition
selected_item = np.array([0, 0])

# Suppose that you interest on a person
# Give me a person which isn't in the dictionnary
image_src = load_image("...")
plt.title('A person to found (not in the dictionnary)')
plt.imshow(image_src)
plt.axis('off')
plt.show()
p_id = get_id_similar_image(image_src)
if p_id == -1:
  print("Similar image not found. Update your image and retry.")
  exit(0)
# Player 2 shall found my person...
selected_item[0] = p_id
selected_item[1] = selected_item[0]
while selected_item[0] == selected_item[1]:
  selected_item[1] = random.choice(range(nb_person))
player_1.reset()
player_2.reset()
selected_question_id_for_player_1 = None
selected_question_id_for_player_2 = None
print('Start board:')
print_whoami(ts.observation[0].numpy())
# Player 1 begin for example
players = cycle([player_1, player_2])
list_q_id = []

try:

  while not ts.is_last():
    player = next(players)
    player_id = 1 if player == player_1 else 2
    other_player_id = 1 + player_id % 2
    action_dict = tf_ttt_env.envs[0].get_current_action()
    # User shall choose the questions
    while True:
        print('Give an english question to play here:')
        input(q)
        q_id = get_id_similar_question(q)
        if q_id == -1:
          print("Similar question not found. Try again.")
        else:
          # Question already given ?
          if q_id in list_q_id:
            print("Question already given. Try another question.")
          else:
            list_q_id.append(q_id)
            selected_question_id_for_player_1 = q_id
            break
    # User will be player 1
    if player_id == 1:
      action_dict['question'][player_id-1] = selected_question_id_for_player_1
    else:
      action_dict['question'][other_player_id-1] = selected_question_id_for_player_1

    player.act()
    ts = tf_ttt_env.current_time_step()
    print('Player:', {player.name}, '\n',
          'Question of player:', question_list[action_dict['question'][player_id-1]], '\n',
          'Answer of other player:', yesno(action_dict['answer'][other_player_id-1]), '\n',
          'Question of other player:', question_list[action_dict['question'][other_player_id-1]], '\n',
          'Answer of player:', yesno(action_dict['answer'][player_id-1]), '\n',
          'Reward:', ts.reward[0].numpy(), 'Board:')
    print_whoami(ts.observation[0].numpy())
    print_seq_questions(tf_ttt_env.envs[0].get_seq_questions())
    print('Your board:')
    id_person_deleted_1 = tf_ttt_env.envs[0].get_deleted_items()
    for i in range(len(id_person_deleted_1)):
      id_person_deleted_1[i] = id_person_deleted_1[i][0]
    display_board(id_person_deleted_1)

except KeyboardInterrupt:
    print('Interrupted by user...')
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

Samples of dictionnary of person generated are given (basic_dict_person.npy and split file extended_dict_person.npy).

Agents models and training checkpoints are also given in checkpoints and model directories.

Note: To unsplit file extended_dict_person.npy, the commands are: cat split_extended_dict_person.tgz_* > extended_dict_person.tgz ; tar xvf extended_dict_person.tgz

## Challenges

Topic doesn't develop a complete model on the "lie factor" to better account for the fact that the adversary may lie to a question.

To be applicable, the volume of data shall be much larger and the model shall be trained extensively on this volume of data (images, questions). We have defined here only an example as proof of concept.

The study of question sequences can be developed on several other themes: Which type of questions are better? Could we define weights for the questions?...    

Anonymous of recorded data used (images) to train the model shall be taken in consideration to deploy a solution with a model computed to recognize an real person in a group with a sequence of questions.   

## What next?

This project could be grow in the development of application on the security domain.

## Acknowledgments

* HOW TO WIN THE GAME “WHO IS IT?”: THANKS TO MATHEMATICS, Pierre-Luc Racine, March 16, 2020, https://urbania.fr/article/comment-gagner-au-jeu-qui-est-ce-grace-aux-mathematiques
* Optimal Strategy in “Guess Who?”: Beyond Binary Search, Mihai Nica, January 19, 2016, https://arxiv.org/pdf/1509.03327
* Multi-Agent Reinforcement Learning with TF-Agents, Dylan Cope, Jun 2020, https://github.com/DylanCope/Multi-Agent-RL-with-TF/blob/master/DMARL%20with%20TF-Agents.ipynb
* Training a Deep Q Network with TF Agents, Tensor Flow, https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=en
