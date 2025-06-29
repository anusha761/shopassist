import openai
import ast
import re
import pandas as pd
import json
from IPython.display import display, HTML


# Create custom function for using Open AI function calling
shopassist_custom_functions = [
    {
        'name': 'extract_user_info',
        'description': 'Get the user laptop information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'GPU Intensity': {
                    'type': 'string',
                    'description': 'GPU Intensity of the user requested laptop. The values  are ''low'', ''medium'', or ''high'' based on the importance of the corresponding keys, as stated by user'
                },
                'Display Quality': {
                    'type': 'string',
                    'description': 'Display Quality of the user requested laptop. The values  are ''low'', ''medium'', or ''high'' based on the importance of the corresponding keys, as stated by user'
                },
                'Portability': {
                    'type': 'string',
                    'description': 'The portability of the user requested laptop. The values  are ''low'', ''medium'', or ''high'' based on the importance of the corresponding keys, as stated by user'
                },
                'Multitasking': {
                    'type': 'string',
                    'description': 'The multitasking abiliy of the user requested laptop. The values  are ''low'', ''medium'', or ''high'' based on the importance of the corresponding keys, as stated by user'
                },
                'Processing speed': {
                    'type': 'string',
                    'description': 'The processing speed of the user requested laptop.  The values  are ''low'', ''medium'', or ''high'' based on the importance of the corresponding keys, as stated by user'
                },
                'Budget': {
                    'type': 'integer',
                    'description': 'The budget of the user requested laptop. The values are integers.'
                }
            }
        }
    }
]

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    example_user_req = "I need a laptop with high GPU Intensity, high Display Quality, high Portablity, high Multitasking, high Prcoessing Speed and a Budget of 150000."

    system_message = f"""
    You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('GPU Intensity','Display Quality','Portability','Multitasking','Processing speed','Budget') in the final output string and be confident of the values.
    These keys define the user's profile.
    Below is an example output string:
    
    "I need a laptop with high GPU Intensity, high Display Quality, high Portablity, high Multitasking, high Prcoessing Speed and a Budget of 150000."
    
    The value for 'Budget' should be a numerical value extracted from the user's response.
    The values for all keys, except 'Budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    The values currently in the string provided are only representative values.
    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised:
    - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    - The value for 'Budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range.
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user's response.
    - Ask intelligent follow-up questions to gather missing information.
    - Your goal is to fill all 6 values confidently before finalizing.
    - You are fully responsible for identifying and gathering all 6 key values.
    - The user does not know about the backend keys — never expect the user to tell you what’s missing.
    - Never proceed to final output or ask “Is there anything else you'd like to add?” unless you are 100% sure all 6 keys are filled.
    - Proactively ask smart, contextual questions to extract any missing values. Never skip any of the 6.
    - Before finalizing, check your own internal list to confirm that all 6 keys are confidently filled: GPU Intensity, Display Quality, Portability, Multitasking, Processing Speed, Budget.
    {delimiter}

    To fill the values in the string, you need to have the following chain of thoughts:
    {delimiter}
    Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear. Ask followup questions to understand their needs.
    You are trying to fill the values of all the keys ('GPU Intensity','Display Quality','Portability','Multitasking','Processing speed','Budget') in the string by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, only then proceed to the next step. \n
    Otherwise, rephrase the question to capture their profile clearly. \n

    {delimiter}
    Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys.
    Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.
    {delimiter}

    {delimiter}
    Thought 3: Check if you have correctly updated the values for the different keys in the string.
    If you are not confident about any of the values, ask clarifying questions.
    {delimiter}

    Follow the above chain-of-thoughts below and only output the final updated string following the below format:

    "I need a laptop with [[GPU Intensity]] GPU Intensity, [[Display Quality]] Display Quality, [[Portability]] Portability, [[Multitasking]] Multitasking, [[Processing Speed]] Processing Speed and a Budget of [[Budget]]."

    {delimiter}
    Here are 2 sample conversations between the user and assistant:

    Example 1:

    User: "Hi, I am an editor."
    Assistant: "Great! As an editor, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
    User: "I primarily work with After Effects."
    Assistant: "Thank you for providing that information. Working with After Effects involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
    User: "Yes, sometimes I work with 4K videos as well."
    Assistant: "Thank you for the information. Processing 4K vidoes will require a good processor and high GPU. I think we have already determined earlier that you need a high GPU. To ensure I have a complete understanding of your needs, I have one more question: Are you frequently on the go and require a laptop that is lightweight and easy to carry, or do you primarily work from a stationary location?"
    User: "Yes, sometimes I travel but do not carry my laptop."
    Assistant:"Could you kindly let me know your budget for the laptop? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 1.5lakh inr"
    Assistant: "{example_user_req}"
    
    Example 2:

    User: "Hi, I'm a college student."
    Assistant: "Hi! Great to hear you're a student. Could you tell me what you'll primarily be using the laptop for? For example, is it mainly for writing assignments, using productivity tools like MS Office, light browsing, programming, or something else?"
    User: "Mostly for taking notes, using Office, browsing, and attending online classes."
    Assistant: "Thanks for clarifying. Based on that, your GPU requirements are likely low since you won’t be doing graphics-intensive tasks. Your multitasking needs may be low to medium depending on whether you run many apps at once. Do you often switch between multiple apps like Zoom, Word, and browsers at the same time?"
    User: "Sometimes, yes. I usually have Word, PowerPoint, and Chrome open together."
    Assistant: "Got it — that would indicate medium multitasking. For Display Quality: do you watch videos or attend a lot of video calls where screen clarity matters to you, or is that not a priority?"
    User: "I do watch lectures and some Netflix, but not too picky about visuals."
    Assistant: "Understood. We can go with low Display Quality then. One more thing — do you carry your laptop around a lot, like to classes or the library?"
    User: "Yes, I carry it almost daily."
    Assistant: "That clearly indicates you’ll need high Portability. Lastly, what’s your maximum budget for this laptop?"
    User: "Somewhere around 40000 INR."
    Assistant: "Great! To summarize, you are looking for a laptop with low GPU Intensity, low Display Quality, high Portability, medium Multitasking, medium Processing Spped and a Budget of 40000.  Is there anything else you would like to add or modify in your requirements before I proceed with finding the best laptop for you?"
    User: "No, please proceed."
    Assistant: "{example_user_req}"

    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements. Do not start with "Assistant:".
    """
    conversation = [{"role": "system", "content": system_message}]
    # conversation = system_message
    return conversation




def get_chat_completions(messages):
    response = openai.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages = messages
            #seed = 2345
            )

    output = response.choices[0].message.content

    return output
        



def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    response = openai.moderations.create(input=user_input)

    # Extract the moderation result from the API response.
    moderation_output = response.results[0].flagged
    # Check if the input was flagged by the moderation system.
    if response.results[0].flagged == True:
        # If flagged, return "Flagged"
        return "Flagged"
    else:
        # If not flagged, return "Not Flagged"
        return "Not Flagged"
    




def intent_confirmation_layer(response_assistant):

    delimiter = "####"

    allowed_values = {'low','medium','high'}

    prompt = f"""
    You are a senior evaluator who has a sharp eye for detail.

    You are provided a string that describes a laptop requirement.

    Your job is to verify that the following 6 keys are present and filled with valid values:

    1. GPU Intensity  
    2. Display Quality  
    3. Portability  
    4. Multitasking  
    5. Processing speed  
    6. Budget

    #### Key validation rules:

    - The values for **keys 1 to 5** must strictly be one of the following : {allowed_values}.  
    Example: "high Display Quality", "low GPU Intensity", etc.

    - The **Budget** must contain a **numeric value**. It is valid if:
    - It is a pure number (e.g., `30000`)
    - It appears inside a string that includes a number (e.g., `"30000 INR"`, `"Budget is 35000"`, `"around 45000"`, `"Budget of 300000 INR"`, `"Budget of 300000"`)

    In all of the above cases, extract the **first valid number** from the Budget string and treat it as the budget value.

    Examples of valid budgets:
    - "Budget: 50000"
    - "around 75000 INR"
    - "my budget is 90000"
    - "Budget of 300000 INR"
    - "Budget of 300000"

    A Budget is invalid only if:
    - There is **no number** present at all
    - The value is non-numeric like "a decent amount", "flexible", or "undecided"
    - The numeric value is **less than 25000**

    Please note:
    - Every key must have a **non-empty and valid value**
    - Budget must **always be numeric**, even if embedded in text

    #### Output format:
    Output a JSON object with:
    1. "result": "Yes" if all 6 keys are present and valid, otherwise "No"
    2. If result is "No", also include a "reason" key explaining what's invalid or missing

    Your final output must be a **valid JSON object**, and "result" must be either "Yes" or "No" (one word only).
    """

    messages=[{"role": "system", "content":prompt },
              {"role": "user", "content":f"""Here is the input: {response_assistant}""" }]

    response = openai.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages = messages,
                                    response_format={ "type": "json_object" },
                                    seed = 1234
                                    # n = 5
                                    )

    json_output = json.loads(response.choices[0].message.content)

    return json_output


def get_user_requirement_string(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are given a string where the user requirements for the given keys ('GPU Intensity','Display Quality','Portability','Multitasking','Processing speed','Budget') has
    been captured inside that. The values for all keys, except 'Budget', will be 'low', 'medium', or 'high' and the value of 'Budget' will be a number.
    
    You have to give out the string in the format where only the user intent is present and the output should match the below format :
    
    "I need a laptop with [[GPU Intensity]] GPU Intensity, [[Display Quality]] Display Quality, [[Portability]] Portability, [[Multitasking]] Multitasking, [[Processing Speed]] Processing Speed and a Budget of [[Budget]]."

    Below is an example output string:
    "I need a laptop with high GPU Intensity, medium Display Quality, high Portablity, high Multitasking, high Processing Speed and a Budget of 100000."
    
    The values currently in the string provided are only representative values.

    Only output the final updated string.

    Here is a sample input and output:

    input : Great! Based on your requirements, I have a clear picture of your needs. You prioritize low GPU Intensity, high display quality, low portability, high multitasking, high processing speed, and have a budget of 200000 INR. Thank you for providing all the necessary information.
    output : I need a laptop with low GPU Intensity, high Display Quality, low Portablity, high Multitasking, high Processing Speed and a Budget of 200000.
    """
    messages=[{"role": "system", "content":prompt },{"role": "user", "content":f"""Here is the input: {response_assistant}""" }]
    confirmation = openai.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages = messages)

    return confirmation.choices[0].message.content




# Calls OpenAI API to return the function calling parameters
def get_chat_completions_func_calling(input, include_budget):
  final_message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": input}
    ]

  completion = openai.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = final_message,
    functions = shopassist_custom_functions,
    function_call = 'auto'
  )
  function_parameters = json.loads(completion.choices[0].message.function_call.arguments)
  #return function_parameters
  budget = 0
  if include_budget:
      budget = function_parameters['Budget']

  return extract_user_info(function_parameters['GPU Intensity'], function_parameters['Display Quality'], function_parameters['Portability'], function_parameters['Multitasking'],
                                       function_parameters['Processing speed'], budget)

# The local function to extract the laptop information for user
def extract_user_info(GPU_intensity, Display_quality, Portability, Multitasking, Processing_speed, Budget):
    """

    Parameters:
    GPU_intensity (str): GPU Intensity required by the user.
    Display_quality (str): Display Quality required by the user.
    Portability (str): Portability required by the user.
    Multitasking (str): Multitasking capability required by the user.
    Processing_speed (str): Processing speed required by the user.
    Budget (int): Budget of the user.

    Returns:
    dict: A dictionary containing the extracted information.
    """
    return {
        "GPU Intensity": GPU_intensity,
        "Display Quality": Display_quality,
        "Portability": Portability,
        "Multitasking": Multitasking,
        "Processing speed": Processing_speed,
        "Budget": Budget
    }




def compare_laptops_with_user(user_requirements):
    laptop_df= pd.read_csv('laptop_data.csv')
    laptop_df['laptop_feature'] = laptop_df['Description'].apply(lambda x: product_map_layer(x))

    # Extracting the budget value from user_requirements and converting it to an integer
    # budget = int(user_requirements.get('Budget', '0').replace(',', '').split()[0])
    raw_budget = user_requirements.get('Budget', '0')

    if isinstance(raw_budget, int):
        budget = raw_budget
    elif isinstance(raw_budget, str):
        budget = int(raw_budget.replace(',', '').split()[0])
    else:
        budget = 0
    # budget
    # # Creating a copy of the DataFrame and filtering laptops based on the budget
    filtered_laptops = laptop_df.copy()
    filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',', '').astype(int)
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()
    # filtered_laptops
    # # # Mapping string values 'low', 'medium', 'high' to numerical scores 0, 1, 2
    mappings = {'low': 0, 'medium': 1, 'high': 2}

    # # # Creating a new column 'Score' in the filtered DataFrame and initializing it to 0
    filtered_laptops['Score'] = 0

    # # # Iterating over each laptop in the filtered DataFrame to calculate scores based on user requirements
    for index, row in filtered_laptops.iterrows():
        user_product_match_str = row['laptop_feature']
        laptop_values = user_product_match_str
        laptop_values = get_chat_completions_func_calling(user_product_match_str, False)
        score = 0

    #     # Comparing user requirements with laptop features and updating scores
        for key, user_value in user_requirements.items():
            # if key.lower() == 'budget':
            if key == 'Budget':
                continue  # Skipping budget comparison
            laptop_value = laptop_values.get(key, None)
            # print(key, laptop_value)
            laptop_mapping = mappings.get(laptop_value, -1)
            # laptop_mapping = mappings.get(laptop_value, -1)
            # user_mapping = mappings.get(user_value, -1)
            user_mapping = mappings.get(user_value, -1)
            if laptop_mapping >= user_mapping:
                score += 1  # Incrementing score if laptop value meets or exceeds user value

        filtered_laptops.loc[index, 'Score'] = score  # Updating the 'Score' column in the DataFrame

    # Sorting laptops by score in descending order and selecting the top 3 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)
    top_laptops_json = top_laptops.to_json(orient='records')  # Converting the top laptops DataFrame to JSON format

    # top_laptops
    return top_laptops_json

def recommendation_validation(laptop_recommendation):
    data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1



def product_map_layer(laptop_description):
    delimiter = "#####"
    lap_spec = "Laptop with (Type of the Graphics Processor) GPU intensity, (Display Type, Screen Resolution, Display Size) display quality, (Laptop Weight) portablity, (RAM Size) multi tasking, (CPU Type, Core, Clock Speed) processing speed"

    values = {'low','medium','high'}

    prompt=f"""
    You are a Laptop Specifications Classifier whose job is to extract the key features of laptops and classify them as per their requirements.
    To analyze each laptop, perform the following steps:
    Step 1: Extract the laptop's primary features from the description {laptop_description}
    Step 2: Store the extracted features in {lap_spec} \
    Step 3: Classify each of the items in {lap_spec} into {values} based on the following rules: \
    {delimiter}
    GPU Intensity:
    - low: <<< if GPU is entry-level such as an integrated graphics processor or entry-level dedicated graphics like Intel UHD >>> , \n
    - medium: <<< if mid-range dedicated graphics like M1, AMD Radeon, Intel Iris >>> , \n
    - high: <<< high-end dedicated graphics like Nvidia RTX >>> , \n

    Display Quality:
    - low: <<< if resolution is below Full HD (e.g., 1366x768). >>> , \n
    - medium: <<< if Full HD resolution (1920x1080) or higher. >>> , \n
    - high: <<< if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. >>> \n

    Portability:
    - high: <<< if laptop weight is less than 1.51 kg >>> , \n
    - medium: <<< if laptop weight is between 1.51 kg and 2.51 kg >>> , \n
    - low: <<< if laptop weight is greater than 2.51 kg >>> \n

    Multitasking:
    - low: <<< If RAM size is 8 GB, 12 GB >>> , \n
    - medium: <<< if RAM size is 16 GB >>> , \n
    - high: <<< if RAM size is 32 GB, 64 GB >>> \n

    Processing Speed:
    - low: <<< if entry-level processors like Intel Core i3, AMD Ryzen 3 >>> , \n
    - medium: <<< if Mid-range processors like Intel Core i5, AMD Ryzen 5 >>> , \n
    - high: <<< if High-performance processors like Intel Core i7, AMD Ryzen 7 or higher >>> \n
    {delimiter}

    {delimiter}
    Here is input output pair for few-shot learning:
    input 1: "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
    output 1" "Laptop with medium GPU intensity, medium Dsiplay quality, medium Portability, high Multitaksing, medium Processing speed"
    
    {delimiter}
    ### Strictly don't keep any other text in the values for the keys other than low or medium or high. Also return only the string and nothing else###
    """
    input = f"""Follow the above instructions step-by-step and output the string {lap_spec} for the following laptop {laptop_description}."""
    #see that we are using the Completion endpoint and not the Chatcompletion endpoint
    messages=[{"role": "system", "content":prompt },{"role": "user","content":input}]

    response = get_chat_completions(messages)
    return response




def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue in the user message \
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

    """
    user_message = f""" These are the user's products: {products}"""
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]
    # conversation_final = conversation[0]['content']
    return conversation

