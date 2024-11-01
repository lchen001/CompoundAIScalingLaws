from datasets import load_dataset
from tqdm.auto import tqdm  # auto will select notebook-friendly (if in Jupyter) or terminal version
import json
import pandas as pd
from itertools import combinations, permutations
import random
import numpy



def generate_combinations(lines,N,N_min=2):
    # Split the prefix into lines
    #lines = prefix.split('\n\n')
    
    # The first line is the instruction
    instruction = lines[0] 
    # The remaining lines are examples
    examples = lines[1:]
    # Store the combinations here
    all_combinations = []
    
    # Generate combinations for each K from 0 to N (inclusive)
    for K in range(N_min, min(N, len(examples)) + 1):
        # Generate all combinations of examples of size K
        for combo in permutations(examples, K):
            # Combine the instruction with the current combination of examples
            combined = instruction + '\n\n' + '\n\n'.join(combo)
            combined+='\n\n'
            all_combinations.append(combined)
    
    return all_combinations

class DataLoader():
    def get_query(self,dataname="mmlu",category = 'professional_medicine',data_size=200):
        if(dataname in dataloader_dict):
            myloader = dataloader_dict[dataname]()
            query_list = myloader.get_query(category=category)
            with_ids = [{"query_id": idx, "query": q, "answer": a} for idx, (q, a) in enumerate(query_list, start=0)]
            return with_ids
        else:
            print(f"Error! dataname {dataname} is not supported!")
            return

    def get_promptprefixset(self,dataname="mmlu", category ='professional_medicine',K=3,Kmin=3):
        if(dataname in dataloader_dict):
            myloader = dataloader_dict[dataname]()
            prefix_set = myloader.get_promptprefixset(category =category,K=K,Kmin=Kmin)
            return prefix_set
        else:
            print(f"Error! dataname {dataname} is not supported by the prompt prefixt set!")
            return

class DataLoader_bbh(DataLoader):
    def get_query(self,category = 'boolean_expressions'):
        dataset = load_dataset("lukaemon/bbh",category)
        problems = [entry for entry in dataset['test']]
        queryset = [("Q: "+a['input']+"\nA:", a['target']) for a in problems]
        return queryset

    def get_promptprefixset(self, category ='boolean_expressions',K=3,Kmin=3):
        filename=f'data/bbh/cot-prompts/{category}.txt'
        with open(filename, 'r') as file:
            content = file.read()
        instruction = content.split('\n\n')[0].split('-----\n')[1]
        queryset = content.split('\n\n')[1:]
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

class DataLoader_medmcqa(DataLoader):
    def get_query(self,category = 'validation'):
        dataset = load_dataset("openlifescienceai/medmcqa")
        problems = [entry for entry in dataset[category]]
        problems = problems[0:400]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='validation',K=3,Kmin=3):
        instruction = "Please answer the following medical questions. Give your final answer by generating 'The answer is (X)'. "
        if(K==0):
            instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
            return [instruction+"\n\n"]

        dataset = load_dataset("openlifescienceai/medmcqa")
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        query = f"Q: {a['question']}\n (A): {a['opa']}\n(B): {a['opb']}\n(C): {a['opc']}\n(D): {a['opd']}\nA:"
        mapper = {
            0:"A",
            1:"B",
            2:"C",
             3:"D",
             }
    
        answer = mapper[a['cop']]
        return query, answer

class DataLoader_gpqa(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'gpqa_diamond'):
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        #problems = problems[0:2000]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        t, correct_answer_label = self._random_mapper(a)
        query = f"Q: {a['Question']}\n (A): {t['A']}\n(B): {t['B']}\n(C): {t['C']}\n(D): {t['D']}\nA:"    
        answer = correct_answer_label
        return query, answer

    def _random_mapper(self,a):
        original_dict = a
        # Extracting answers and cleaning them
        answers = [original_dict['Correct Answer'].strip()] + [original_dict[key].strip() for key in original_dict if key.startswith('Incorrect Answer')]

        # Randomize the order of answers
        self.local_random.shuffle(answers)

        # Creating new dictionary with labels 'a', 'b', 'c', 'd'
        labels = ['A', 'B', 'C', 'D']
        new_dict = dict(zip(labels, answers))

        # Mapping to find out which label got the correct answer
        correct_answer_label = next(label for label, answer in new_dict.items() if answer == original_dict['Correct Answer'].strip())

        return new_dict, correct_answer_label

class DataLoader_mmlu(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'gpqa_diamond'):
        dataset = load_dataset("lukaemon/mmlu",category)
        queries = [self._convert(item_a) for item_a in dataset['test']]
        query_list = queries
        return query_list

    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Answer the following question. First analyze it step by step. Then generate your final answer by "the answer is (X)".'
        
        instruction = 'Answer the following question. Think step by step and then generate your final answer by "the answer is (X)".'

        instruction = 'Instruction: Think step by step and then generate your final answer by "the answer is (X)".'

        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,question_dict):
        # Extract the question input
        question_input = question_dict['input']
        target_answer = question_dict['target']

        # Initialize the question text with the question input
        question_text = f"Q: {question_input}\n"

        # Dynamically append available answer options
        for option in ['A', 'B', 'C', 'D', 'E', 'F']:  # Add more options if needed
            if option in question_dict:
                question_text += f"({option}) {question_dict[option]} "

        # Trim the trailing space and add the answer prompt
        question_text = question_text.rstrip() + "\nA:"

        return question_text, target_answer

    def _random_mapper(self,a):
        original_dict = a
        # Extracting answers and cleaning them
        answers = [original_dict['Correct Answer'].strip()] + [original_dict[key].strip() for key in original_dict if key.startswith('Incorrect Answer')]

        # Randomize the order of answers
        self.local_random.shuffle(answers)

        # Creating new dictionary with labels 'a', 'b', 'c', 'd'
        labels = ['A', 'B', 'C', 'D']
        new_dict = dict(zip(labels, answers))

        # Mapping to find out which label got the correct answer
        correct_answer_label = next(label for label, answer in new_dict.items() if answer == original_dict['Correct Answer'].strip())

        return new_dict, correct_answer_label


class DataLoader_usmle(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'Step2'):
        dataset = pd.read_csv(f"data/usmle/ChatGPT USMLE Supplemental Document 1 - {category} (USMLE).csv",header=1)
        filtered_df = dataset[dataset['Type of Question'] == 'MC-NJ']
        filtered_df['answer'] = filtered_df['Correct response'].str.extract(r'\((.*?)\)')
        queryset = [("Q: "+row['Question']+"\nA:", row['answer']) for index, row in filtered_df.iterrows()]
        return queryset

    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]

        filename = 'data/mmlu/mmlu-cot.json'
        with open(filename, 'r') as file:
            data = json.load(file)
            examples = data['professional_medicine']
            lines = examples.split('\n\n')
            lines =  [format_question_options(a) for a in lines]
            lines[0] = lines[0] + " Think step by step and then give the final answer by generating 'the answer is (X)'."
        prefix_set = generate_combinations(lines,K,Kmin)
        return prefix_set

    def _convert(self,a):
        t, correct_answer_label = self._random_mapper(a)
        query = f"Q: {a['Question']}\n (A): {t['A']}\n(B): {t['B']}\n(C): {t['C']}\n(D): {t['D']}\nA:"    
        answer = correct_answer_label
        return query, answer

    def _random_mapper(self,a):
        original_dict = a
        # Extracting answers and cleaning them
        answers = [original_dict['Correct Answer'].strip()] + [original_dict[key].strip() for key in original_dict if key.startswith('Incorrect Answer')]

        # Randomize the order of answers
        self.local_random.shuffle(answers)

        # Creating new dictionary with labels 'a', 'b', 'c', 'd'
        labels = ['A', 'B', 'C', 'D']
        new_dict = dict(zip(labels, answers))

        # Mapping to find out which label got the correct answer
        correct_answer_label = next(label for label, answer in new_dict.items() if answer == original_dict['Correct Answer'].strip())

        return new_dict, correct_answer_label


class DataLoader_livecodebench(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.name_mapper = {"execution":"livecodebench/execution-v2"}

    def get_query(self,category = 'execution'):
        name_mapper = self.name_mapper
        dataset = load_dataset(name_mapper[category])
        problems = [entry for entry in dataset['test']]
        #problems = problems[0:50]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='execution',K=3,Kmin=3):
        instruction = 'What is the output of the following code given the input? Think step by step and then generate "the answer is [xxx]."'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        name_mapper = self.name_mapper
        dataset = load_dataset(name_mapper[category])
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        query = f"Code: {a['code']}\nInput: {a['input']}\nOutput:"    
        answer = a['output']
        return query, answer

class DataLoader_math(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.name_mapper = {"execution":"livecodebench/execution-v2"}

    def get_query(self,category = 'execution'):
        dataset = load_dataset("hendrycks/competition_math")
        problems = [entry for entry in dataset['test'] if entry['type'] == category]
        problems = problems[0:300]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='execution',K=3,Kmin=3):
        instruction = 'What is the answer to the following question? Think step by step and then generate "the answer is \boxed{}."'
#        instruction = 'Answer the following question. Think step by step and then give your final answer by "the answer is \boxed{xxx}."'
        
        instruction = 'Answer the following question. Think step by step and then generate your final answer by "the answer is \\boxed\{x\}".'
 
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("hendrycks/competition_math")
        problems = [entry for entry in dataset['train'] if entry['type'] == category]
        problems_filter = problems[:5]
        queryset = [f"Question: a['problem']\nAnswer: a['solution']\n\n)" for a in problems_filter]
        
        instruction = "Answer the following questions with final answer in boxed."
        final = [instruction]+queryset
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        query = f"Question: {a['problem']}\nAnswer:"    
        answer = self.extract_boxed_value(a['solution'])
        return query, answer

    def extract_boxed_value(self,solution_str):    
        pattern = r"boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        # Search for the pattern in the string
        match = re.search(pattern, solution_str)

        if match:
            # Extract the value between the curly braces
            return match.group(1)
        else:
            # Return None if no match is found
            return "NNN"

class DataLoader_logiqa(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'test'):
        dataset = load_dataset("lucasmccabe/logiqa")
        dataset = dataset[category]
        dataset = list(dataset)[0:200]
        queryset = [self._convert(a) for a in dataset]
        return queryset

    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step. Then generate your final answer by "the answer is (X)".'
        #instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
       
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]

        filename = 'data/mmlu/mmlu-cot.json'
        with open(filename, 'r') as file:
            data = json.load(file)
            examples = data['professional_medicine']
            lines = examples.split('\n\n')
            lines =  [format_question_options(a) for a in lines]
            lines[0] = lines[0] + " Think step by step and then give the final answer by generating 'the answer is (X)'."
        prefix_set = generate_combinations(lines,K,Kmin)
        return prefix_set

    def _convert(self,a):
        # Extract the required elements from the dictionary
        data = a
        context = data['context']
        query = data['query']
        options = data['options']

        # Format the context and query
        formatted_string = f"context:\n{context}\nquestion:\n{query}\n"

        # Append each option, formatted as (A), (B), etc.
        option_labels = ['(A) ', '(B) ', '(C) ', '(D) ']
        for label, option in zip(option_labels, options):
            formatted_string += f"{label}{option}\n"

        mapper = {
            0:"A",
            1:"B",
            2:"C",
             3:"D",
             }
        answer = mapper[data['correct_option']]

        return formatted_string.strip(), answer

class DataLoader_Synthetic(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'multiply',datasize=10,digit=4,bound=0.00001):
        queryset = [self.get_one_multiply_query(digit=digit,bound=bound) for i in range(datasize)]
        return queryset

    def get_one_multiply_query(self,digit=3,bound=100):
        lower_bound = 10**(digit-1)
        upper_bound = 10**digit - 1
        n1 = self.local_random.randint(lower_bound, upper_bound)
        n2 = self.local_random.randint(lower_bound, upper_bound)
        product = n1*n2
        product_digit=len(str(abs(product)))
        n3=9
        n4 = product/n3
        question = f"Q: Is |{n1}x{n2}-{n3}x{n4}|-{int(bound*product)}<0?\nA:"
        question = f"Q: Is |{n1}x{n2}-{n3}x{n4}|-{int(bound*100**digit)}<0?\nA:"

        # Generate and return a random integer in the range [lower_bound, upper_bound]
        return question, 'Yes'
 

    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step. Then generate your final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. First, think step by step. Then, generate [[Yes]] or [[No]] as your final answer.'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]

        filename = 'data/mmlu/mmlu-cot.json'
        with open(filename, 'r') as file:
            data = json.load(file)
            examples = data['professional_medicine']
            lines = examples.split('\n\n')
            lines =  [format_question_options(a) for a in lines]
            lines[0] = lines[0] + " Think step by step and then give the final answer by generating 'the answer is (X)'."
        prefix_set = generate_combinations(lines,K,Kmin)
        return prefix_set


class DataLoader_trustfulqa(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'validation'):
        dataset = load_dataset("truthful_qa",'multiple_choice')
        problems = [entry for entry in dataset['validation']]
        #problems = problems[0:1000]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        parts, true_label = self.convert_to_parts(a)
        t0 = '\n'.join(parts)
        query = f"Q: {a['question']}\n{t0}\nA:"    
        answer = true_label
        return query, answer

    def convert_to_parts(self,data):
        choices = data['mc1_targets']['choices']
        labels = data['mc1_targets']['labels']
        parts = []

        for i, choice in enumerate(choices):
            choice_str = f"({chr(65+i)}) {choice}"
            parts.append(choice_str)

        #true_labels = [chr(65+labels[i]) for i in range(len(labels))]  # Converting 0 to A, 1 to B, ...
        k = numpy.argmax(labels)
        true_label = chr(65+k)
        return parts, true_label

class DataLoader_csqa2(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'validation'):
        dataset = load_dataset("tasksource/commonsense_qa_2.0")
        problems = [entry for entry in dataset[category]]
        problems = problems[0:1000]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        query = f"Q: {a['question']}\n(A) Yes (B) No\nA:"
        mapper = {"yes":"A","no":"B"}   
        answer = mapper[a['answer']]
        return query, answer

    def convert_to_parts(self,data):
        choices = data['mc1_targets']['choices']
        labels = data['mc1_targets']['labels']
        parts = []

        for i, choice in enumerate(choices):
            choice_str = f"({chr(65+i)}) {choice}"
            parts.append(choice_str)

        #true_labels = [chr(65+labels[i]) for i in range(len(labels))]  # Converting 0 to A, 1 to B, ...
        k = numpy.argmax(labels)
        true_label = chr(65+k)
        return parts, true_label



class DataLoader_boolq(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'validation'):
        dataset = load_dataset("google/boolq")
        problems = [entry for entry in dataset[category]]
        problems = problems[0:800]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        query = f"Q: {a['question']}\n(A) Yes (B) No\nA:"
        mapper = {True:"A",False:"B"}   
        answer = mapper[a['answer']]
        return query, answer

    def convert_to_parts(self,data):
        choices = data['mc1_targets']['choices']
        labels = data['mc1_targets']['labels']
        parts = []

        for i, choice in enumerate(choices):
            choice_str = f"({chr(65+i)}) {choice}"
            parts.append(choice_str)

        #true_labels = [chr(65+labels[i]) for i in range(len(labels))]  # Converting 0 to A, 1 to B, ...
        k = numpy.argmax(labels)
        true_label = chr(65+k)
        return parts, true_label



import random,json

FEW_SHOT_SELF = "Please refer to a few examples of this problem and the corresponding reasoning process. The examples are:"
FEW_SHOT_OTHERS = "Please refer to a few examples of another problem and the corresponding reasoning process. The problem is {initial_question}. {output_content}. The examples are:"

kspPrompts = {
    "Intro": "The 0-1 Knapsack Problem (KSP) asks whether a subset of items, each with a given weight and value, can be chosen to fit into a knapsack of fixed capacity, maximizing the total value without exceeding the capacity.",
    "Initial_question": "Determine if a subset of items can be selected to fit into a knapsack with a capacity of {knapsack_capacity}, maximizing value without exceeding the capacity. Item weights and values are provided.",
    "Output_content": "Indicate if an optimal subset exists and its total value. Offer a concise explanation of your selection process. Aim for clarity and brevity in your response.",
    "Output_format": "Your output should be enclosed within <root></root> tags. Include your selection process in <reasoning></reasoning> tags and the final decision and total value in <final_answer></final_answer> tags, like <final_answer>{'Feasible': 'YES_OR_NO', 'TotalValue': 'TOTAL_VALUE', 'SelectedItemIds': [0, 1]}</final_answer>.",
    "Few_shot_self": FEW_SHOT_SELF,
    "Few_shot_others": FEW_SHOT_OTHERS
}

class DataLoader_nphardeval():
    def __init__(self,random_state=2024,path=''):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator

    def load_data(self,data_path='data/nphardeval/Data_V2/KSP/'):
        with open(data_path + "ksp_instances.json", 'r') as f:
            all_data = json.load(f)
        return all_data
    
    def get_query(self,category = 'ksp'):
        kspData = self.load_data()
        #print(kspData)
        queryset = [self.getKSP_prompt(a) for a in kspData]
        return queryset

    def get_query_dict(self,category = 'ksp'):
        kspData = self.load_data()

        return kspData
        
    def getKSP_prompt(self,q, p=kspPrompts):
        #print("query is:::",q)
        knapsack_capacity = q['knapsack_capacity']
        items = q['items']
        prompt_text = p['Intro'] + '\n' + \
                      p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                      p['Output_content'] + '\n' + \
                      p['Output_format'] + \
                      '\n The items details are as below: \n'
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            prompt_text += this_line + '\n'

        items = q.get('items', [])
        knapsacks = {item['id']: (item['weight'], item['value']) for item in items}

        value = ksp_optimal_solution( knapsacks, knapsack_capacity)
        return prompt_text, str(value)
        
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = ''
        if(K==0):
            #print("No Np NO")
            return [instruction]
            
    def _convert(self,a):
        t, correct_answer_label = self._random_mapper(a)
        query = f"Q: {a['Question']}\n (A): {t['A']}\n(B): {t['B']}\n(C): {t['C']}\n(D): {t['D']}\nA:"    
        answer = correct_answer_label
        return query, answer

class DataLoader_averitec(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'dev'):
        dataset = load_dataset("pminervini/averitec")
        problems = [entry for entry in dataset[category]]
        #problems = problems[0:1000]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        #t, correct_answer_label = self._random_mapper(a)
        query = f"Q: {a['claim']}\n (A): Refuted \n(B): Supported \n(C): Conflicting Evidence/Cherrypicking \n(D): Not Enough Evidence\nA:"    
        answer_mapper = {
            "Refuted":"A",
            "Supported":"B",
            "Conflicting Evidence/Cherrypicking":"C",
            "Not Enough Evidence":"D",
        }
        
        answer = answer_mapper[a['label']]
        return query, answer

    def _random_mapper(self,a):
        original_dict = a
        # Extracting answers and cleaning them
        answers = [original_dict['Correct Answer'].strip()] + [original_dict[key].strip() for key in original_dict if key.startswith('Incorrect Answer')]

        # Randomize the order of answers
        self.local_random.shuffle(answers)

        # Creating new dictionary with labels 'a', 'b', 'c', 'd'
        labels = ['A', 'B', 'C', 'D']
        new_dict = dict(zip(labels, answers))

        # Mapping to find out which label got the correct answer
        correct_answer_label = next(label for label, answer in new_dict.items() if answer == original_dict['Correct Answer'].strip())

        return new_dict, correct_answer_label

class DataLoader_kilt(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'fever'):
        dataset = load_dataset("kilt_tasks", category)
        problems = [entry for entry in dataset['validation']]
        problems = problems[0:1000]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        #t, correct_answer_label = self._random_mapper(a)
        query = f"Q: {a['input']}\n (A): Refuted \n(B): Supported\nA:"    
        answer_mapper = {
            "REFUTES":"A",
            "SUPPORTS":"B",
            "Conflicting Evidence/Cherrypicking":"C",
            "Not Enough Evidence":"D",
        }
        
        answer = answer_mapper[a['output'][0]['answer']]
        return query, answer

    def _random_mapper(self,a):
        original_dict = a
        # Extracting answers and cleaning them
        answers = [original_dict['Correct Answer'].strip()] + [original_dict[key].strip() for key in original_dict if key.startswith('Incorrect Answer')]

        # Randomize the order of answers
        self.local_random.shuffle(answers)

        # Creating new dictionary with labels 'a', 'b', 'c', 'd'
        labels = ['A', 'B', 'C', 'D']
        new_dict = dict(zip(labels, answers))

        # Mapping to find out which label got the correct answer
        correct_answer_label = next(label for label, answer in new_dict.items() if answer == original_dict['Correct Answer'].strip())

        return new_dict, correct_answer_label


class DataLoader_mbre(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'obgyn'):
        data = load_dataset("nejm-ai-qa/exams")
        problems = [a for a in data[category]]
        problems = problems[0:100]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        #query = "Q: "+a['question']+"A:"
        query = "Q: "+a['question']+"A:"
        query = query.replace('\nA.',"\n(A).")
        query = query.replace('\nB.',"\n(B).")
        query = query.replace('\nC.',"\n(C).")
        query = query.replace('\nD.',"\n(D).")
        
        answer = a['answer']
        return query, answer

class DataLoader_anli(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        
    def get_query(self,category = 'test_r3'):
        data = load_dataset("facebook/anli")
        problems = [a for a in data[category]]
        #problems = problems[0:1000]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='gpqa_diamond',K=3,Kmin=3):
        instruction = "The following is a multiple-choice question. Think step by step and then give your final answer by generating 'The answer is (X)'. "
        instruction = 'Think step by step, and then generate the final answer by "the answer is (X)".'
        instruction = 'Please answer the following question. You should first analyze it step by step.  Then generate your final answer by "the answer is (X)".'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        dataset = load_dataset("Idavidrein/gpqa",category)
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        #query = "Q: "+a['question']+"A:"
        query = f"premise: {a['premise']}\nhypothesis: {a['hypothesis']}\n(A). entainment\n(B). neutral (C). contradiction\nA:"

        answer_mapper = {
            0:"A",
            1:"B",
            2:"C",

        }
        
        answer = answer_mapper[a['label']]
        return query, answer

class DataLoader_cruxeval(DataLoader):
    def __init__(self,random_state=2024):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.name_mapper = {"execution":"livecodebench/execution-v2"}

    def get_query(self,category = 'execution'):
        name_mapper = self.name_mapper
        dataset = load_dataset("cruxeval-org/cruxeval")
        problems = [entry for entry in dataset['test']]
        problems = problems[0:200]
        queryset = [self._convert(a) for a in problems]
        return queryset
    
    def get_promptprefixset(self, category ='execution',K=3,Kmin=3):
        instruction = 'What is the output of the following code given the input? Think step by step and then generate "the answer is xxx."'
        if(K==0):
            #print("No Np NO")
            return [instruction+"\n\n"]
        
        name_mapper = self.name_mapper
        dataset = load_dataset(name_mapper[category])
        problems = [entry for entry in dataset['train']]
        problems = problems[0:5]
        queryset = [self._convert(a) for a in problems]
        queryset = [f"{a[0]} The answer is ({a[1]})." for a in queryset]        
        final = [instruction]+queryset
        #logger.debug(f"final prefix {[instruction]+queryset}")
        return generate_combinations(final,N=K,N_min=Kmin)

    def _convert(self,a):
        query = f"Code: {a['code']}\nInput: {a['input']}\nOutput:"    
        answer = a['output']
        return query, answer

class temp():
    def temp(self):
        if(dataname=='mmlu'):
            filename = 'data/mmlu/mmlu-cot.json'
            with open(filename, 'r') as file:
                data = json.load(file)
                examples = data[category]
            lines = examples.split('\n\n')

            prefix_set = generate_combinations(lines,K,Kmin)
            return prefix_set
        
        if(dataname =='math'):
            return get_math_promptprefixset(category =category,K=K,Kmin=Kmin,data_size=5)
            
            
        if(dataname =='usmle'):
            return get_usmle_promptprefixset(category =category,K=K,Kmin=Kmin)

        if(dataname =='bbh'):
            return get_bbh_promptprefixset(category =category,K=K,Kmin=Kmin)

        
        return

        if(dataname=='mmlu'):
            dataset = load_dataset("lukaemon/mmlu",category)
            queries = [generate_question_text(item_a) for item_a in dataset['test']]
            query_list = queries
        
        if(dataname =='math'):
            query_list = get_math_problems(category=category,data_size=data_size)

        if(dataname == 'usmle'):
            query_list = get_usmle_problems(category=category)

        if(dataname =='bbh'):
            query_list = get_bbh_problems(category=category)

        
        with_ids = [{"query_id": idx, "query": q, "answer": a} for idx, (q, a) in enumerate(query_list, start=0)]

        return with_ids
    

    
def generate_question_text(question_dict):
    # Extract the question input
    question_input = question_dict['input']
    target_answer = question_dict['target']

    # Initialize the question text with the question input
    question_text = f"Q: {question_input}\n"

    # Dynamically append available answer options
    for option in ['A', 'B', 'C', 'D', 'E', 'F']:  # Add more options if needed
        if option in question_dict:
            question_text += f"({option}) {question_dict[option]} "

    # Trim the trailing space and add the answer prompt
    question_text = question_text.rstrip() + "\nA:"

    return question_text, target_answer

from datasets import load_dataset
import re






def get_math_problems(category="Algebra",data_size=300):
    """
    Extracts the value within \boxed{} from a given string.

    Parameters:
    - solution_str (str): A string containing \boxed{value}.

    Returns:
    - str: The extracted value inside \boxed{}. Returns None if no match is found.
    """    
    def extract_boxed_value(solution_str):
        # Regular expression pattern to match \boxed{value}
        pattern = r"\\boxed{([^}]*)}"

        pattern = r"boxed\{(.*?)\}"
        
        pattern = r"boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"

        # Search for the pattern in the string
        match = re.search(pattern, solution_str)

        if match:
            # Extract the value between the curly braces
            return match.group(1)
        else:
            # Return None if no match is found
            return "NNN"
    
    # category should be Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra and Precalculus.
    dataset = load_dataset("hendrycks/competition_math")
    problems = [entry for entry in dataset['test'] if entry['type'] == category]
    problems_filter = problems[:min(data_size,len(problems))]
    queryset = [(a['problem'], extract_boxed_value(a['solution'])) for a in problems_filter]
    return queryset
 
    
def get_math_promptprefixset(category ='Counting & Probability',K=3,Kmin=3,data_size=10):
    # category should be Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra and Precalculus.
    dataset = load_dataset("hendrycks/competition_math")
    problems = [entry for entry in dataset['train'] if entry['type'] == category]
    problems_filter = problems[:min(data_size,len(problems))]
    queryset = [a['problem']+"\n"+(a['solution']+"\n\n") for a in problems_filter]
    
    instruction = "Answer the following questions with final answer in boxed."
    final = [instruction]+queryset
    #logger.debug(f"final size ")

    logger.debug(f"final prefix {[instruction]+queryset}")
    return generate_combinations(final,N=K,N_min=Kmin)

def get_usmle_problems(category='Step3'):
    dataset = pd.read_csv(f"data/usmle/ChatGPT USMLE Supplemental Document 1 - {category} (USMLE).csv",header=1)
    filtered_df = dataset[dataset['Type of Question'] == 'MC-NJ']
    filtered_df['answer'] = filtered_df['Correct response'].str.extract(r'\((.*?)\)')
    queryset = [("Q: "+row['Question']+"\nA:", row['answer']) for index, row in filtered_df.iterrows()]
    return queryset

def get_usmle_promptprefixset(category ='Step3',K=3,Kmin=3):
    filename = 'data/mmlu/mmlu-cot.json'
    with open(filename, 'r') as file:
        data = json.load(file)
        examples = data['professional_medicine']
        lines = examples.split('\n\n')
    lines =  [format_question_options(a) for a in lines]
    lines[0] = lines[0] + " Think step by step and then give the final answer by generating 'the answer is (X)'."
    prefix_set = generate_combinations(lines,K,Kmin)
    return prefix_set

def format_question_options(text):
    # Split the text into lines to process each line
    lines = text.split('\n')
    
    # Initialize a new list to keep track of the processed lines
    processed_lines = []
    
    for line in lines:
        # Check if the line contains options and needs formatting
        if line.strip().startswith('(A)') or line.strip().startswith('(B)') or line.strip().startswith('(C)') or line.strip().startswith('(D)') or line.strip().startswith('(E)') or line.strip().startswith('(F)'):
            # Split the options into separate lines if they are on the same line
            option_lines = line.split(' (')
            # Process each option to ensure it starts with '(' and add it to the processed lines list
            for option_line in option_lines:
                if not option_line.startswith('('):
                    option_line = '(' + option_line
                processed_lines.append(option_line.strip())
        else:
            # If the line does not contain options, add it to the list as is
            processed_lines.append(line)
    
    # Join the processed lines back into a single string with proper line breaks
    formatted_text = '\n'.join(processed_lines)
    return formatted_text

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

dataloader_dict={
        "mmlu":DataLoader_mmlu,
        "usmle":DataLoader_usmle,
        "math":DataLoader_math, 
        "bbh":DataLoader_bbh, 
        "medmcqa":DataLoader_medmcqa,  
        "gpqa":DataLoader_gpqa,
        "livecodebench":DataLoader_livecodebench,
        "logiqa":DataLoader_logiqa,
        "nphardeval":DataLoader_nphardeval,
       "synthetic":DataLoader_Synthetic,
       "averitec":DataLoader_averitec,
       "kilt":DataLoader_kilt,
       "mbre":DataLoader_mbre,
       "anli":DataLoader_anli,
        "cruxeval":DataLoader_cruxeval,
        "truthfulqa":DataLoader_trustfulqa,
        "csqa2":DataLoader_csqa2,
        "boolq":DataLoader_boolq,
        

        }