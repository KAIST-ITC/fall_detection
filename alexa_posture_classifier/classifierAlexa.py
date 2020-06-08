from flask import Flask
from flask_ask import Ask, statement, question, session
import globalVariable

app = Flask(__name__)
ask = Ask(app, "/")


@ask.launch
def start():
    speechText = "Welcome to the pose classifier"
    return statement(speechText)


@ask.intent("answerIntent")
def answerFunction():
    pose = globalVariable.pose
    speechText = 'You are ' + str(pose)
    return statement(speechText)


@ask.intent("AMAZON.FallbackIntent")
def fallBackFunction():
    speechText = 'I was not able to understand you'
    return statement(speechText)


@ask.intent("AMAZON.HelpIntent")
def helpFunction():
    speechText = 'You can say something like ... ask classifier whether I am sitting'
    return statement(speechText)