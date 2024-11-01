import re

def extract_answer(text,method="mc_majority_vote"):
    patterns = {
           "mc_majority_vote":r"answer is \((.*?)\)", 
           "mc_noparenthesis_majority_vote":r"answer is (\w)", 
           "boxed_majority_vote":r"boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
           "answeris_majority_vote":r"answer is (.+?)\.",
          "yesno_majority_vote":r'\b(yes|no)\b',
        }

    patterns_func = {
           "ksp_majority_vote":parse_xml_to_dict,
        }



    patterns2 = {
           "mc_majority_vote":r"\(([A-Fa-f])\)", 
        }

    if(method in patterns_func):
        try:
            output1, reasoning = patterns_func[method](text)
            output1 = output1['TotalValue']
        except Exception as e:
            output1 = "NNN"
        return output1


    if method in patterns:
        pattern = patterns[method]
    else:
        print("Invalid extract_answer method!")

    # If a match is found, return the captured group, otherwise return None
    match = re.search(pattern, text.lower())
    if match:
        return match.group(1)  # The first captured group, which is "X"
    else:
        return "NNN"

        #pattern2 = r"\([A-Fa-f]\)"
        pattern2 = patterns2[method]
        match2 = re.search(pattern2, text)
        if match2:
            #print("--match 2--",match2.group)
            return match2.group(1)
        else:
            return "NNN"

    # Define the regular expression pattern to match "answer is (X)"
    pattern = r"\\boxed{([^}]*)}"
    pattern = r"boxed\{(.*?)\}"    

    pattern = r"boxed\{((?:\\.|[^{}])*(?:\{(?:\\.|[^{}])*\}(?:\\.|[^{}])*)*)\}"

    pattern = r"answer is \((.*?)\)"
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    

def extract_idx(text):
    # Define the regular expression pattern to find "the answer id is {idx}"
    pattern = r"best answer id is (\d+)"
    
    # Search the text for the pattern
    match = re.search(pattern, text)
    
    # If a match is found, convert the captured group (idx) to an integer and return it
    if match:
        return int(match.group(1))
    else:
        # If no match is found, you might want to handle it differently
        # For this example, returning None to indicate no index was found
        #print(f"WARNING: no id found in {text}")
        return 0

def isinvalid(text):
    if text=="NNN":
        return True
    else:
        return False



import xml.etree.ElementTree as ET
import ast
import os
#import path
#from path import MODEL_TYPE_PATH, HP_HARD_PATH

def append_root_tags(string):
    if not string.strip().startswith("<root>"):
        string = "<root>\n" + string
    if not string.strip().endswith("</root>"):
        string += "\n</root>"
    return string

def parse_xml_to_dict(xml_string: str):
    """Parse the XML string to a dictionary.

    :param xml_string: The XML string to parse.
    :return: A tuple of (output, reasoning).
    """
    # Append root tags if necessary
    #print(xml_string)
    xml_string = append_root_tags(xml_string)

    # remove comments
    remove_comment_func = lambda string: string.split('//')[0].rstrip() if '//' in string else string
    xml_string = '\n'.join(remove_comment_func(line) for line in xml_string.split('\n'))
    
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find the 'final_answer' tag
    final_answer_element = root.find('final_answer')

    # Find the 'reasoning' tag
    reasoning = root.find('reasoning').text.strip()

    # Convert the 'final_answer' tag to a dictionary
    output = ast.literal_eval(final_answer_element.text.strip())
    # print(reasoning_element.text)
    return output, reasoning


def convert_q_to_q1_ksp(q):
    # Extract knapsack capacity from the string
    capacity_match = re.search(r'capacity of (\d+)', q)
    knapsack_capacity = int(capacity_match.group(1)) if capacity_match else 0
    
    # Extract item details using regex
    items = re.findall(r'Item (\d+) has weight (\d+) and value (\d+)', q)
    items_list = [{'id': int(id), 'weight': int(weight), 'value': int(value)} for id, weight, value in items]
    
    # Construct the dictionary in the desired format
    q1_format = {
        'items': items_list,
        'knapsack_capacity': knapsack_capacity
    }
    return q1_format


def kspCheck(q, output):
    """Validates the solution for the KSP instance.

    :param instance: A dictionary of the KSP instance.
    :param solution: A dictionary of the solution.
    :return: A tuple of (is_correct, message).
    """

    solution, reasoning = parse_xml_to_dict(output)

    instance = convert_q_to_q1_ksp(q)
    # Change string key to integer key and value to boolean
    items = instance.get('items', [])
    knapsacks = {item['id']: (item['weight'], item['value']) for item in items}

    ksp_optimal_value = ksp_optimal_solution(knapsacks, instance['knapsack_capacity'])

    is_feasible = (solution.get('Feasible', '').lower() == 'yes')
    if is_feasible != (ksp_optimal_value > 0):
        return False, f"The solution is {is_feasible} but the optimal solution is {ksp_optimal_value > 0}."
    
    total_value = int(solution.get('TotalValue', -1))
    selectedItems = list(map(int, solution.get('SelectedItemIds', [])))

    if len(set(selectedItems)) != len(selectedItems):
        return False, f"Duplicate items are selected."

    total_weight = 0
    cum_value = 0

    # Calculate total weight and value of selected items
    for item in selectedItems:
        if knapsacks.get(item, False):
            weight, value = knapsacks[item]
            total_weight += weight
            cum_value += value
        else:
            return False, f"Item {item} does not exist."

    # Check if the item weight exceeds the knapsack capacity
    if total_weight > instance['knapsack_capacity']:
        return False, f"Total weight {total_weight} exceeds knapsack capacity {instance['knapsack_capacity']}."

    if total_value != cum_value:
        return False, f"The total value {total_value} does not match the cumulative value {cum_value} of the selected items."

    '''
    if total_value != ksp_optimal_value:
        return False, f"The total value {total_value} does not match the optimal value {ksp_optimal_value}."
    '''

    return True, f"The solution is valid with total weight {total_weight} and total value {total_value}."



def ksp_optimal_solution(knapsacks, capacity):
    """Provides the optimal solution for the KSP instance with dynamic programming.
    
    :param knapsacks: A dictionary of the knapsacks.
    :param capacity: The capacity of the knapsack.
    :return: The optimal value.
    """
    num_knapsacks = len(knapsacks)

    # Create a one-dimensional array to store intermediate solutions
    dp = [0] * (capacity + 1)

    for itemId, (weight, value) in knapsacks.items():
        for w in range(capacity, weight - 1, -1):
            dp[w] = max(dp[w], value + dp[w - weight])

    return dp[capacity]
