import openai
import ast
import re
import pandas as pd
import json
from IPython.display import display, HTML
from flask import Flask, redirect, url_for, render_template, request
from functions import *

openai.api_key = open('OPENAI_API_Key.txt', 'r').read().strip()

app = Flask(__name__)

conversation_bot = []

conversation = initialize_conversation()
introduction = get_chat_completions(conversation)
conversation_bot.append({'bot': introduction})
top_3_laptops = None


@app.route("/")
def default_func():
    global conversation_bot, conversation, top_3_laptops
    return render_template("index.html", name_xyz = conversation_bot)

@app.route("/end_conv", methods = ['POST', 'GET'])
def end_conv():
    global conversation_bot, conversation, top_3_laptops
    conversation_bot = []
    conversation = initialize_conversation()
    introduction = get_chat_completions(conversation)
    conversation_bot.append({'bot': introduction})
    top_3_laptops = None
    return redirect(url_for('default_func'))

@app.route("/invite", methods = ['POST'])
def invite():
    global conversation_bot, conversation, top_3_laptops, conversation_reco
    user_input = request.form["user_input_message"]
    prompt = 'Remember that you are a intelligent laptop shopping assistant. You should help and answer only with the queries related to laptops. If the queries are not related to laptops, just say something like you can help only with queries related to laptops etc.'
    moderation = moderation_check(user_input)
    if moderation == 'Flagged':
        display("Sorry, this message has been flagged. Please restart your conversation.")
        return redirect(url_for('end_conv'))
    
    if top_3_laptops is None:

        conversation.append({"role": "user", "content": user_input + prompt})
        conversation_bot.append({'user': user_input})

        response_assistant = get_chat_completions(conversation)
        moderation = moderation_check(response_assistant)
        if moderation == 'Flagged':
            display("Sorry, this message has been flagged. Please restart your conversation.")
            return redirect(url_for('end_conv'))    


        
        print("Response_assistant: ",response_assistant)
        
        confirmation = intent_confirmation_layer(response_assistant)

        print("Intent Confirmation Yes/No:",confirmation.get('result'))

        if "No" in confirmation.get('result'):
            print("Reason:",confirmation.get('reason'))
            conversation.append({"role": "assistant", "content": str(response_assistant)})
            conversation_bot.append({'bot': response_assistant})

        else:
            print(response_assistant)
            response = get_user_requirement_string(response_assistant)
            print("response:",response)
            result = get_chat_completions_func_calling(response, True)
            print("result:",result)
            print("WAIT")
            conversation_bot.append({'bot': "Thank you for providing all the information. Kindly wait, while I fetch the products: \n"})

            top_3_laptops = compare_laptops_with_user(result)

            print("top 3 laptops are", top_3_laptops)

            validated_reco = recommendation_validation(top_3_laptops)
            if len(validated_reco) == 0:
                conversation_bot.append({'bot': "Sorry, we do not have laptops that match your requirements."})

            conversation_reco = initialize_conv_reco(validated_reco)
            conversation_reco.append({"role": "user", "content": "This is my user profile" + str(validated_reco)})
            recommendation = get_chat_completions(conversation_reco)

            moderation = moderation_check(recommendation)
            if moderation == 'Flagged':
                display("Sorry, this message has been flagged. Please restart your conversation.")
                return redirect(url_for('end_conv'))

            conversation_reco.append({"role": "assistant", "content": str(recommendation)})
            conversation_bot.append({'bot': recommendation})

    else:
        conversation_reco.append({"role": "user", "content": user_input})
        conversation_bot.append({'user': user_input})

        response_asst_reco = get_chat_completions(conversation_reco)

        moderation = moderation_check(response_asst_reco)
        if moderation == 'Flagged':
            print("Sorry, this message has been flagged. Please restart your conversation.")
            return redirect(url_for('end_conv'))

        conversation.append({"role": "assistant", "content": response_asst_reco})
        conversation_bot.append({'bot': response_asst_reco})

    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

    