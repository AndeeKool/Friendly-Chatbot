#!/usr/bin/env python
# coding: utf-8
"""Using the parser to recognise your own semantics

spaCy's parser component can be trained to predict any type of tree
structure over your input text. You can also predict trees over whole documents
or chat logs, with connections between the sentence-roots used to annotate
discourse structure. In this example, we'll build a message parser for a common
"chat intent": finding local businesses. Our message semantics will have the
following types of relations: ROOT, PLACE, QUALITY, ATTRIBUTE, TIME, LOCATION.

"show me the best hotel in berlin"
('show', 'ROOT', 'show')
('best', 'QUALITY', 'hotel') --> hotel with QUALITY best
('hotel', 'PLACE', 'show') --> show PLACE hotel
('berlin', 'LOCATION', 'hotel') --> hotel with LOCATION berlin

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
TRAIN_DATA = [
    (
        #0  1
        "hi there",
        {
            "heads": [0, 0],  # index of token head
            "deps": ["ROOT", "-"],
        },
    ),
    (
        #0   1
        "hey you",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0     1
        "hello bot",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0    1
        "good morning",
        {
            "heads": [1, 1],
            "deps": ["QUALITY", "ROOT"],
        },
    ),
    (
        #0
        "hi",
        {
            "heads": [0],
            "deps": ["ROOT"],
        },
    ),
    (
        #0   1   2
        # are -> is -> Situacion actual, estado, sentimientos, etc
        "how are you",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "STATE", "TARGET"],
        },
    ),
    (
        #0   1   2   3
        "how are you feeling",
        {
            "heads": [0, 2, 0, 2],
            "deps": ["ROOT", "STATE", "TARGET", "STATE"],
        },
    ),
    (
        #0   1   2   3
        "how are you doing?",
        {
            "heads": [0, 2, 0, 2, 0],
            "deps": ["ROOT", "STATE", "TARGET", "STATE", "QUESTION"],
        },
    ),
    (
        #0   1   2
        "how you doing",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "TARGET", "STATE"],
        },
    ),
    (
        #0   1      2
        "I'm feeling sad",
        {
            "heads": [3, 0, 3, 3],
            "deps": ["TARGET", "STATE", "-", "ROOT"],
        },
    ),

    (
        #0   1      2
        "oh I feel anxious now",
        {
            "heads": [0, 3, 1, 3, 3],
            "deps": ["-", "TARGET", "STATE", "ROOT", "TIME"],
        },
    ),

    (
        #0   1      2
        "I'm so happy today",
        {
            "heads": [3, 0, 3, 3, 3],
            "deps": ["TARGET", "STATE", "-", "ROOT", "TIME"],
        },
    ),
    (
        #0   1   2
        "I'm very angry",
        {
            "heads": [3, 0, 3, 3],
            "deps": ["TARGET", "STATE", "-", "ROOT"],
        },
    ),
    (
        #0   1      2       3       4   5
        "I'm very stressed because of school",
        {
            "heads": [2, 2, 2, 5, 5, 2, 2],
            "deps": ["TARGET", "STATE", "-", "ROOT", "-", "-", "REASON"],
        },
    ),
    (
        #0   1  2   3   4
        "My job is very stressful",
        {
            "heads": [4, 0, 4, 4, 4],
            "deps": ["TARGET", "REASON", "STATE", "-", "ROOT"],
        },
    ),
    (
        #0  1  2    3
        "I'm not good",
        {
            "heads": [3, 0, 3, 3],
            "deps": ["TARGET", "STATE", "ATTRIBUTE", "ROOT"],
        },
    ),
    (
        #0 1   2    3   4       5
        "I'm feeling a little down",
        {
            "heads": [5, 0, 5, 4, 5, 5],
            "deps": ["TARGET", "STATE", "STATE", "-", "-", "ROOT"],
        },
    ),
    (
        #0   1   2  3   4
        "Today is a bad day",
        {
            "heads": [0, 3, 3, 4, 4],
            "deps": ["TIME", "STATE", "-","ROOT","TARGET"],
        },
    ),
    (
        #0          1
        "Goodbye chatbot",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0   1   2  3   4
        "Bye talk to you later",
        {
            "heads": [0, 0, 3, 1, 1],
            "deps": ["ROOT", "STATE", "-", "TARGET", "TIME"],
        },
    ),
    (
        #   0      1    2
        "Farewell my friend",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "-","TARGET"],
        },
    ),
    (
        #0   1   2 
        "Byebye bot",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0   1   2 
        "Sayonara bot",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "TARGET"],
        },
    ),
    (
        #0 1     2  3   4   5
        "I'm going to sleep goodnight",
        {
            "heads": [4, 0, 4, 4, 5, 5],
            "deps": ["TARGET", "STATE", "-","-", "STATE", "ROOT"],
        },
    ),
    
    # (
    #     #0    1 2    3    4     5
    #     "find a cafe with great wifi",
    #     {
    #         "heads": [0, 2, 0, 5, 5, 2],  # index of token head
    #         "deps": ["ROOT", "-", "PLACE", "-", "QUALITY", "ATTRIBUTE"],
    #     },
    # ),
    # (
    #     #0    1 2     3    4   5
    #     "find a hotel near the beach",
    #     {
    #         "heads": [0, 2, 0, 5, 5, 2],
    #         "deps": ["ROOT", "-", "PLACE", "QUALITY", "-", "ATTRIBUTE"],
    #     },
    # ),
    # (
    #     "find me the closest gym that's open late",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 6, 4, 6, 6],
    #         "deps": [
    #             "ROOT",
    #             "-",
    #             "-",
    #             "QUALITY",
    #             "PLACE",
    #             "-",
    #             "-",
    #             "ATTRIBUTE",
    #             "TIME",
    #         ],
    #     },
    # ),
    # (
    #     "show me the cheapest store that sells flowers",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 4, 4, 4],  # attach "flowers" to store!
    #         "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "-", "PRODUCT"],
    #     },
    # ),
    # (
    #     "find a nice restaurant in london",
    #     {
    #         "heads": [0, 3, 3, 0, 3, 3],
    #         "deps": ["ROOT", "-", "QUALITY", "PLACE", "-", "LOCATION"],
    #     },
    # ),
    # (
    #     "show me the coolest hostel in berlin",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 4, 4],
    #         "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "LOCATION"],
    #     },
    # ),
    # (
    #     "find a good italian restaurant near work",
    #     {
    #         "heads": [0, 4, 4, 4, 0, 4, 5],
    #         "deps": [
    #             "ROOT",
    #             "-",
    #             "QUALITY",
    #             "ATTRIBUTE",
    #             "PLACE",
    #             "ATTRIBUTE",
    #             "LOCATION",
    #         ],
    #     },
    # ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            # print("Losses", losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


def test_model(nlp):
    texts = [
        "hello bot",
        "hello there",
        "hi good morning",
        "hey bot",
        "Hello",
        "HI THERE",

        # "Toaster",
        # "Water bottle",
        # "Buy things",

        "how are you doing bot",
        "how do you do",
        "how do you feel",
        "I'm feeling sad",
        "I'm sad",

        # "how is the weather",
        # "how did the cat get there",
        # "how can I find the restroom",

        "hi my name is Steve",

        "I'm very happy today",
        "I'm great",
        "I feel sad",

        
        # "find a hotel with good wifi",
        # "find me the cheapest gym near work",
        # "show me the best hotel in berlin",
    ]
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
        "You can tell me about your day if you wish.",
        "How is your day going?",
        "Any news from the real world?",
        "Is there anything you want to tell me?",
    ]
    questions = ["how"]
    targets_self = ["bot", "you", "chatbot"]
    self_state_responses = [
        "I'm doing fine thank you.",
        "Thanks for asking, I'm doing alright.",
        "Right now I'm feeling great! Just a little sleepy.",
    ]

    targets_user = ["I","i","'m", "me", "my", "myself"]

    good_state = ["good", "amazing","happy", "excited", "exciting", "love", "great"]
    bad_state = ["angry", "stress", "stressed", "stressful", "sad", "down", "not good", "bad", "wrong", "anxious", "hard"]

    happy_responses = [
        "That's good to hear!",
        "I'm glad.",
        "Oh that's amazing! I'm so happy for you!",
        'That sounds great!',
        "That's cool.",
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

    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != "-"])
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

        # Dependency label dictionary
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

        print("\n")
        print("-" * 20)
        print("\n")


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # find a hotel with good wifi
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('hotel', 'PLACE', 'find'),
    #   ('good', 'QUALITY', 'wifi'),
    #   ('wifi', 'ATTRIBUTE', 'hotel')
    # ]
    # find me the cheapest gym near work
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('cheapest', 'QUALITY', 'gym'),
    #   ('gym', 'PLACE', 'find'),
    #   ('near', 'ATTRIBUTE', 'gym'),
    #   ('work', 'LOCATION', 'near')
    # ]
    # show me the best hotel in berlin
    # [
    #   ('show', 'ROOT', 'show'),
    #   ('best', 'QUALITY', 'hotel'),
    #   ('hotel', 'PLACE', 'show'),
    #   ('berlin', 'LOCATION', 'hotel')
    # ]