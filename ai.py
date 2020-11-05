import spacy
import random

greetings = ["hi", "hello", "hey", "morning", "afternoon", "yo"]
greeting_responses = [
    "Hi!",
    "Hello friendly human.",
    "Hi there!",
    "Hey!",
]
welcome_responses = [
    "Hi there! I'm a bot and you can say hi to me.",
    "Hello! I'm a greeting bot.",
    "Welcome, feel free to say hi to me anytime.",
    "Hey human! I'm a bot, but you can say hi to me and I'll do my best to try and answer.",
]
questions = ["how"]
targets_self = ["bot", "you", "chatbot"]
self_state_responses = [
    "I'm doing fine thank you.",
    "Thanks for asking, I'm doing alright.",
    "Right now I'm feeling great! Just a little sleepy.",
]

targets_user = ["I'm", "I", "i", "me", "my", "myself"]

good_state = ["good", "amazing","happy", "excited", "exciting", "love"]
bad_state = ["angry", "stress", "stressed", "stressful", "sad", "down", "not good", "bad", "wrong", "anxious", "hard"]

happy_responses = [
    "That's good to hear!",
    "I'm glad.",
    "Oh that's amazing! I'm so happy for you!",
    'That sounds great!',
]

cheerup_responses = [
    "Everything is going to be fine!",
    "Don't worry, you can do this!",
    "It may be hard, but I know you can do it!",
    "Keep it up! I believe in you!",
    "Don't be too harsh with yourself",
    "Take a step back and breath, everything will work out!",
]

farewells = ["goodbye", "bye", "byebye", "bye-bye", "farewell", "sayonara", "goodnight"]

farewell_responses = [
    "Goodbye human friend, have a nice day!",
    "Bye, let's talk again soon!",
    "See you later!",
    "Farewell human!",
    "Are you leaving so soon? Goodbye!",
    "It was nice talking to you! Bye!",

]

class AI():


    def __init__(self):
        self.nlp = spacy.load('model')

    def message(self, msg):
        if not msg:
            return None

        doc = self.nlp(msg)

        label_dict = {t.dep_ : t for t in doc}
        print(f"label_dict: {label_dict}")

        responses = []

        # Check if ROOT is a known greeting
        if label_dict["ROOT"].text.lower() in greetings:
            # Answer with random greeting
            responses += [random.choice(greeting_responses)]
        elif label_dict["ROOT"].text.lower() in questions:
            # Is a question
            # Answer iif user asks how I am
            if "STATE" in label_dict and label_dict["TARGET"].text.lower() in targets_self:
                responses += [random.choice(self_state_responses)]
            else:
                responses += ["I'm sorry, I'm not sure how to answer that."]
        # If user talks about their feelings
        elif "STATE" in label_dict and label_dict["TARGET"].text.lower() in targets_user:
            if label_dict["ROOT"].text.lower() in good_state:
                responses += [random.choice(happy_responses)]
            elif label_dict["ROOT"].text.lower() in bad_state:
                responses += [random.choice(cheerup_responses)]
        elif label_dict["ROOT"].text.lower() in farewells:
            responses += [random.choice(farewell_responses)]
        else:
            # Answer with a random welcome message
            responses += [random.choice(welcome_responses)]

        print("Response:")
        print(responses)

        return ' '.join(responses)