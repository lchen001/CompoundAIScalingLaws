import numpy, random, sys
from util import extract_answer, extract_idx, isinvalid, kspCheck, parse_xml_to_dict
import os
sys.path.append('../service/')
sys.path.append('service/')
from llmengine import LLMEngine
from modelservice import GenerationParameter
model_list=['openaichat/gpt-4-0125-preview','openaichat/gpt-4-1106-preview','openaichat/gpt-4','openaichat/gpt-4-0314','openaichat/gpt-4-0613','openaichat/gpt-3.5-turbo','openaichat/gpt-3.5-turbo-0125','openaichat/gpt-3.5-turbo-1106','openaichat/gpt-3.5-turbo-0301','openaichat/gpt-3.5-turbo-0613',
           'anthropic/claude-2','azure_openai/GPT-4-0613-32K','azure_openai/GPT-4-0613','azure_openai/GPT-35-0613','azure_openai/GPT-35-0301','llama2chat/70b']

db_path = os.getenv('DB_PATH', 'db/db_test.sqlite')

MyLLMEngine = LLMEngine(service_name=model_list,
                 #db_path="db/db_azure.sqlite"
                 db_path=db_path,                        
                       )

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from concurrent.futures import ProcessPoolExecutor

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a file handler which logs even debug messages
fh = logging.FileHandler('my_log.log')
fh.setLevel(logging.DEBUG)

# Create a console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


class InferNetwork(object):
    def __init__(self,
                 gen_name='openaichat/gpt-3.5-turbo-0125',
                 gen_prompt_prefix=[''],
                 gen_n=5,
                 gen_temp=1,
                 gen_temp_min=0,
                 gen_temp_max=1,
                 gen_ids=[0],

                 enhance_name="openaichat/gpt-3.5-turbo-0125",
                 enhance_layer_n = 0,
                 enhance_temp = 0.1,
                 enhance_instruction='First analyze the question step by step. Then give the final answer by generating  "answer is (X)".',

                 use_filter=False,
                 filter_name = "openaichat/gpt-3.5-turbo-0125",
                 filter_temp  = 0.7,
                 judge_type = 'multi_step',

                 selector_name='openaichat/gpt-3.5-turbo-0125',
                 selector_n=10,
                 selector_mode = 'pairwise',
                 selector_id = 0,

                 random_state=2024,
                 get_full=False,
                 max_workers=100,
                 random_state_order=0,
                 metric = 'mc_majority_vote',
                ):
        #gen_temp=1
        self.random_state = random_state
        self.random_state_order = random_state_order
        self.get_full=False
        self.max_workers = max_workers
        self.metric = metric
        self.use_filter=use_filter
        
        # init the generators
        self.generators = []
        self.gen_prompt_prefix = gen_prompt_prefix
        self.iter = 0
        self.gen_ids = gen_ids
        self.gen_temp = gen_temp
        self.gen_name = gen_name
        self.gen_temp_min = gen_temp_min
        self.gen_temp_max = gen_temp_max
        # init the enhancer 
        self.enhancers = []
        self.enhance_layer_n=enhance_layer_n
        self.enhance_instruction = enhance_instruction


        # init the filter
        self.filter = Filter(s_id=0,name=filter_name,temp=filter_temp,judge_type=judge_type)

        if(len(gen_prompt_prefix)<gen_n):
            logger.info(f"Warning! The number of prefix {len(gen_prompt_prefix)} is smaller than the number of generators {gen_n}")
        local_random = random.Random()  # Create a new random generator instance
        local_random.seed(self.random_state)  # Seed the local generator
        for i in range(gen_n):
            idx = local_random.randint(0, len(self.gen_prompt_prefix)-1)
            prompt_prefix = self.gen_prompt_prefix[idx]
            idx2 = local_random.randint(0, len(self.gen_ids)-1)
            gen_id = self.gen_ids[idx2]
            temp_i = local_random.uniform(gen_temp_min, gen_temp_max)
            temp_i = self.gen_temp
            gen1 = Generator(temp=temp_i,name=gen_name,s_id=gen_id,prompt_prefix=prompt_prefix)
            self.generators.append(gen1)
        self.gen_temp_list = []
        for i in range(len(gen_ids)):
            temp_i = local_random.uniform(gen_temp_min, gen_temp_max)
            temp_i = self.gen_temp
            self.gen_temp_list.append(temp_i)
        #for i in range(enhance_layer_n):
        enhance1 = Enhancer(name=enhance_name,temp=enhance_temp,s_id=0)
        self.enhancers = enhance1
        # init the selectors 
        self.selector = Selector(name=selector_name,s_id=selector_id,method=selector_mode,metric=self.metric)
        return
    
    def respond(self,query):
        #responses = [self.generators[i].respond(query) for i in range(len(self.generators))]
        responses = self._parallel_generate(query)

 
        for i in range(self.enhance_layer_n):
            #responses = [self.enhancers.enhance(instruction=self.enhance_instruction,answer=t,query=query) for idx, t in enumerate(responses)]        
            responses = self._parallel_enhance_responses(responses, query)

        # use filter
        if(self.use_filter==True):
            responses = self.filter.filter(query,responses)
        num_filtered = len(responses)

        local_random = random.Random()  # Create a new random generator instance
        local_random.seed(self.random_state_order)  # Seed the local generator
        local_random.shuffle(responses)

        final_out = self.selector.choose(query,responses)
        if(self.get_full==True):
            return {"answer":final_out['choice'],"generator_responses":final_out['count'],"raw_answer":final_out['raw_answer'],"full_responses":responses}        
        final_out.update({"answer":final_out['choice'],"generator_count":final_out['mc_count'],"raw_answer":final_out['raw_answer'],"gen_n":len(self.generators),"enhance_layer_n":self.enhance_layer_n,'num_filtered':num_filtered})
        full_res = self.get_answer_poll(query)
        final_out.update(full_res)
        return final_out
        
    def get_answer_poll(self,query):
        '''
        raw_responses = []
        for g_id in self.gen_ids:
            for prefix in self.gen_prompt_prefix:
                gen1 = Generator(temp=self.gen_temp,name=self.gen_name,s_id=g_id,prompt_prefix=prefix)
                res = gen1.respond(query)
                raw_responses.append(res)
        '''
        raw_responses = self._parallel_generate_answer_pull(query)

        for i in range(self.enhance_layer_n):
            #responses = [self.enhancers.enhance(instruction=self.enhance_instruction,answer=t,query=query) for idx, t in enumerate(responses)]        
            raw_responses = self._parallel_enhance_responses(raw_responses, query)

        raw_responses_new = [item for item in raw_responses]

        final_answers = [extract_answer(a,method=self.metric) for a in raw_responses_new]
        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1/len(final_answers)
            else:
                count[item]+=1/len(final_answers)

        responses = {"full_possible_answer_count_before_filter":count}

        if(self.use_filter==True):
            raw_responses = self.filter.filter(query,raw_responses)

        final_answers = [extract_answer(a,method=self.metric) for a in raw_responses]
        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1/len(final_answers)
            else:
                count[item]+=1/len(final_answers)


        responses.update({"full_possible_answer_poll":raw_responses,"full_possible_answer_count":count})
        
        # use filter
        if(self.use_filter==True):
            filter_stats = self.get_answer_poll_filter_stats(query,raw_responses_new)
            responses.update(filter_stats)

            raw_responses = self.filter.filter(query,raw_responses)

        return responses

    def get_answer_poll_filter_stats(self,query,raw_responses):
        raw_responses1 = self.filter.filter(query,raw_responses)
        final_answers = [extract_answer(a,method=self.metric) for a in raw_responses1]
        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1/len(final_answers)
            else:
                count[item]+=1/len(final_answers)

        # get "z" and "w"
        final_answers_raw = [extract_answer(a,method=self.metric) for a in raw_responses]
        z = []
        for idx, item in enumerate(final_answers_raw):
            z.append(item)
        w = []
        for idx, item in enumerate(final_answers_raw):
            if(raw_responses[idx] in raw_responses1):
                w.append(1)
            else:
                w.append(0)
        
        

        responses = {"filtered_possible_answer_count":count,"filter_totalremain":len(raw_responses),"z":z,"w":w}
        
        return responses

    def _parallel_generate_answer_pull(self, query):
        # Helper function to be executed in parallel
        def generate_response(g_id, prefix):
            gen = Generator(temp=self.gen_temp_list[g_id], name=self.gen_name, s_id=g_id, prompt_prefix=prefix)
            return gen.respond(query)

        # Create a list to store futures
        future_responses = []
        
        # Use ThreadPoolExecutor to parallelize the requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create tasks for each combination of g_id and prefix
            for g_id in self.gen_ids:
                for prefix in self.gen_prompt_prefix:
                    # Submit each task to the executor
                    future = executor.submit(generate_response, g_id, prefix)
                    future_responses.append(future)
        
        raw_responses = [future.result() for future in future_responses]
        return raw_responses

    def _parallel_generate(self, query):
        # Function to be executed in parallel for each generator
        def get_response(generator):
            return generator.respond(query)
        # Using ThreadPoolExecutor to execute the get_response function in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Mapping the get_response function over self.generators
            # executor.map returns an iterator that yields results as they are completed
            responses = list(executor.map(get_response, self.generators))
        return responses

    def _parallel_enhance_responses(self, responses, query):
        # Helper function to be used with ThreadPoolExecutor
        def enhance_response(response):
            return self.enhancers.enhance(instruction=self.enhance_instruction, answer=response, query=query)

        # Use ThreadPoolExecutor to execute enhancements in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            enhanced_responses = list(executor.map(enhance_response, responses))
        
        return enhanced_responses


class Generator:
    def __init__(self,
                name='openaichat/gpt-4-0125-preview',
                s_id=0,
                temp=0.1,
                prompt_prefix="",
                ):
        self.trial=s_id
        self.model=name
        self.temp=temp
        self.prompt_prefix = prompt_prefix
        return
    
    def respond(self,
                query):
        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=self.trial,
        )
        res = MyLLMEngine.getcompletion(query=self.prompt_prefix+query,service_name=self.model,genparams=genparams)
        return res


class Selector:
    def __init__(self,
                 name='openaichat/gpt-4-1106-preview',
                 s_id=0,
                 method="mc_majority_vote",
                 temp=0.01,
                 max_workers=100,
                 metric='mc_majority_vote',

                 ):
        self.trial=s_id
        self.model=name
        self.method=method
        self.temp=temp
        self.max_workers=max_workers
        self.metric = metric
        return
    
    def choose(self,query,answers):
        actions={
           "mc_majority_vote":self.choose_mc_majority_vote, 
           "mc_llm_full_select":self.choose_mc_llm_full_select,
           "mc_llm_pairwise_select":self.choose_mc_llm_pairwise_select,
            "mc_llm_pairwiseunique_select":self.choose_mc_llm_pairwiseunique_select,
        }

        if self.method in actions:
            return actions[self.method](query, answers)
        else:
            print("Invalid selector method!")

    def choose_mc_majority_vote(self,query,answers):
        final_answers = [extract_answer(a,method=self.metric) for a in answers]
        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1
            else:
                count[item]+=1
        max_key = max(count, key=count.get)
        for idx, item in enumerate(final_answers):
            if item == max_key:
                raw_answer = answers[idx]
                choic_id = idx
        return {"choice_id":choic_id,"mc_count":count,"choice":max_key,"raw_answer":raw_answer}
        
    def choose_mc_llm_full_select(self,query,answers):
        # 1. Format the prompt with all candidate answers 
        prompt = "Below is a user question and a few candidate answers. Read them carefully and select the answer with the highest accuracy and correctness. Start by analyzing the quality of each answer carefully. Do not be affected by the order of all answers. Then give your final selection by 'the best answer id is i', where i is the answer id.\n"                
        prompt+='[user query]\n{query}\n'.format(query=query)
        # add all answers
        final_answers = [extract_answer(a,method=self.metric) for a in answers]
        for idx, a in enumerate(answers):
            prompt+='[start of candidate answer with id {i}]\n{answer}\n[end of candidate answer with id {i}]\n'.format(i=idx,answer=a)
        prompt+='[your analysis and selection]'
        # 2. query the model
        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=self.trial,
        )
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams)
        logger.debug(f"query is{query}")
        logger.debug(f"the input prompt is{prompt}")
        logger.debug(f"selector response {res}")
        # 3. extract the final id
        choice_id = extract_idx(res)
        max_key = final_answers[extract_idx(res)]
        raw_answer = answers[choice_id]

        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1
            else:
                count[item]+=1
        return {"choice_id":choice_id,"mc_count":count,"choice":max_key,"raw_answer":raw_answer,"raw_judgement":res}

    def choose_mc_llm_pairwise_select(self,query,answers):
        # 1. get pairwise comparison
        final_answers = [extract_answer(a,method=self.metric) for a in answers]
        scores, explanations = self._parallel_choose_mc_llm_pairwise_select(query=query,answers=answers)
        scores = self.refine_score(scores,final_answers)
        '''
        scores = numpy.zeros((len(answers),len(answers)))
        for i in range(len(answers)):
            for j in range(len(answers)):
                if(i==j):
                    continue
                label = self.pairwise(query, (answers[i], answers[j]))
                if(label==0):
                    scores[i,j]+=1    
                if(label==1):
                    scores[j,i]+=1
        '''

        id_max_sum = numpy.argmax(numpy.sum(scores, axis=1))
        # 2. extract the final id
        choice_id = id_max_sum
        final_answers = [extract_answer(a,method=self.metric) for a in answers]

        max_key = final_answers[id_max_sum]
        raw_answer = answers[choice_id]
        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1
            else:
                count[item]+=1
        return {"choice_id":choice_id,"mc_count":count,"choice":max_key,"raw_answer":raw_answer,"scores":scores,"explanations":explanations,"final_answers":final_answers,"raw_answers":answers}

    def choose_mc_llm_pairwiseunique_select(self,query,answers):
        # 1. get pairwise comparison
        final_answers = [extract_answer(a,method=self.metric) for a in answers]
        scores, explanations = self._parallel_choose_mc_llm_pairwise_select(query=query,answers=answers)
        scores = self.refine_score(scores,final_answers)
        # 2. map the score to a unique score matrix
        unique_answers, scores_unique_answers = self.map_score(scores, final_answers)
        # 3. choose the final answer with the best overall perf
        id_max_sum_raw = numpy.argmax(numpy.sum(scores_unique_answers, axis=1))
        # 4. return one raw answer with the same final answer
        for idx,a in enumerate(final_answers):
            if (a==unique_answers[id_max_sum_raw]):
                id_max_sum = idx
                break

        # 5. extract the final id
        choice_id = id_max_sum
        final_answers = [extract_answer(a,method=self.metric) for a in answers]

        max_key = final_answers[id_max_sum]
        raw_answer = answers[choice_id]
        count = dict()
        for item in final_answers:
            if (item not in count):
                count[item]=1
            else:
                count[item]+=1
        return {"choice_id":choice_id,"mc_count":count,"choice":max_key,"raw_answer":raw_answer,"scores":scores,"explanations":explanations,"final_answers":final_answers,"raw_answers":answers,"scores_unique_answers":scores_unique_answers,"unique_answers":unique_answers}


    def _parallel_choose_mc_llm_pairwise_select(self, query, answers):
        scores = numpy.zeros((len(answers), len(answers)))
        explanations = numpy.zeros((len(answers), len(answers)),dtype=object)


        # Define a function to be executed in parallel
        def process_pair(i, j):
            if i == j:
                return None  # No need to process pairs of the same index
            labels = self.pairwise(query, (answers[i], answers[j]))
            #print("labels are------",labels)
            label, explanation = labels
            return (i, j, label, explanation)

        # Use ThreadPoolExecutor to parallelize pairwise comparisons
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(process_pair, i, j) for i in range(len(answers)) for j in range(len(answers))]

            # Collect and process results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    i, j, label, exp = result
                    explanations[i,j] = exp
                    if label == 0:
                        scores[i, j] += 0.5
                    elif label == 1:
                        scores[j, i] += 0.5
        return scores, explanations

    def pairwise(self,query, answers):
        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=0.01,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=self.trial,
        )
        # give all answers to a model
        # 
        #prompt = "Please act as an impartial judge and evaluate the quality of two answers to the user question displayed below. Begin your evaluation by comparing the two answers and provide a short explanation. You evaluation should focus on accuracy and correctness. Do not be affected by the order of these answers. Do not pick plausible but false answers. Give your final selection by 'the better answer id is i', where i is the answer id.\n"
        #prompt = "Below is a user question and two candidate answers. Read them carefully and select the answer with the highest accuracy and correctness. Start by analyzing the quality of each answer carefully. Do not be affected by the order of all answers. Then give your final selection by 'the best answer id is i', where i is the answer id.\n"
        prompt = "Below is a user question and two candidate answers. Read them carefully and select the  answer with the highest accuracy and correctness. Start by analyzing the quality of each answer carefully. Do not be affected by the order of all answers. Then give your final selection by 'the best answer id is i', where i is the answer id.\n"
        #prompt = "Please act as an impartial judge and evaluate the quality of two answers to the user question displayed below. Do not be affected by the order of these answers. Begin your evaluation by comparing the two answers and provide a short explanation. You evaluation should focus on accuracy and correctness. Give your final selection by 'the better answer id is i', where i is the answer id.\n"
        prompt = "Below is a user question and two candidate answers. Read them carefully and select the answer with the highest accuracy and correctness. Start by comparing the quality of the given answers critically. Do not be affected by the order of the candidate answers. Then give your final selection by 'the best answer id is i', where i is the answer id.\n"                

        prompt+='[start of user question]\n{query}\n[end of user question]\n'.format(query=query)
        # add all answers
        final_answers = [extract_answer(a,method=self.metric) for a in answers]
        if(final_answers[0] == final_answers[1]):
            return 0, "SAME FINAL ANSWER"

        for idx, a in enumerate(answers):
            prompt+='[start of candidate answer with id {i}]\n{answer}\n[end of candidate answer with id {i}]\n'.format(i=idx,answer=a)
        prompt+='[your analysis and selection]'
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams)
        #print("res is------",res)
        return extract_idx(res), res        

    def refine_score(self,scores,final_answers):
        for i in range(len(final_answers)):
            for j in range(len(final_answers)):
                if(final_answers[i]==final_answers[j]):
                    scores[i,j] = 0.01
                    scores[j,i] = 0.01
                if(isinvalid(final_answers[i])):
                    scores[i,j] = 0
                if(i==j):
                    scores[i,j] = 0
        return scores

    def map_score(self,scores,final_answers):
        unique_answers = sorted(set(final_answers))
        scores_unique_answers = numpy.zeros((len(unique_answers),len(unique_answers)))
        counts_unique_answers = numpy.zeros((len(unique_answers),len(unique_answers)))

        unique_answers_map = {}
        for i in range(len(final_answers)):
            for j in range(len(unique_answers)):
                if(final_answers[i]==unique_answers[j]):
                    unique_answers_map[i]=j    

        for i in range(len(final_answers)):
            for j in range(len(final_answers)):
                idx_i = unique_answers_map[i]
                idx_j = unique_answers_map[j]
                if(idx_i==idx_j):
                    scores_unique_answers[idx_i,idx_j]=0.5
                else:
                    counts_unique_answers[idx_i,idx_j]+=2
                    counts_unique_answers[idx_j,idx_i]+=2
                    if(scores[i,j]==1):
                        scores_unique_answers[idx_i,idx_j]+=2
                    if(scores[i,j]==0):
                        scores_unique_answers[idx_j,idx_i]+=2
                    if(scores[i,j]==0.5):
                        scores_unique_answers[idx_i,idx_j]+=1
                        scores_unique_answers[idx_j,idx_i]+=1
       
        for i in range(len(unique_answers)):
            for j in range(len(unique_answers)):
                if(i!=j):
                    scores_unique_answers[i,j] = scores_unique_answers[i,j]/counts_unique_answers[i,j]

        return unique_answers, scores_unique_answers


class Enhancer:
    def __init__(self,
                 name='openaichat/gpt-4-1106-preview',
                 s_id=0,
                 temp=0.1,

                 ):
        self.trial=s_id
        self.model=name
        self.temp=temp
        return

    def enhance(self,
                query,
                instruction,
                answer,
                ):
        return self.enhance_two_step(query=query,instruction=instruction,answer=answer)

        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=self.trial,
        )
        return self.enhance_two_step(query,instruction,answer)
        # give all answers to a model
        prompt = f"Here is a question and a candidate's answer. Please (i) criticize the candidate answer, starting with [Analysis], and (ii) then generate a new answer to the original question based on your analysis, starting with [Enhanced Answer].  Make sure your new answer follow the given format instruction.\n"
        prompt += f"[Format Instruction]\n{instruction}\n[User Question]{query}\n[Candidate Answer]{answer}"
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams)

        parts = res.split("[Enhanced Answer]")
        # The second part of the split is what comes after "[Enhanced Answer]"
        newanswer = parts[1] if len(parts) > 1 else ""

        return newanswer

    def enhance_two_step(self,
                query,
                instruction,
                answer,
                ):

        # step 1: get analysis
        genparams1 = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=self.trial,
        )

#        prompt = f"Here is a question and a candidate answer.\nPlease analyze the candidate answer very critically. Focus on logical or factual mistakes. You will have a 20 dollar tip if you do a great job.\n"
        prompt = f"You are an impartial and smart expert in criticizing answer quality. \nPlease criticize the given answer to the user question below. Focus your analysis on logical or factual mistakes.\n"

#        prompt = f"You are an impartial and intelligent expert in analyzing answer quality. \nPlease analyze the given answer to the user question critically. Focus your analysis on logical or factual mistakes.\n"

        prompt = f"You are an impartial and intelligent expert in analyzing answer quality. \nPlease analyze the given answer to the user question critically. Focus your analysis on logical, factual, and deduction mistakes.\n"

        prompt = f"You are an impartial and intelligent expert in judging answer quality. \nPlease judge the given answer to the user question very critically. Focus your judgement on logical, factual, and deduction mistakes.\n"

        prompt += f"[User Question]{query}\n[Given Answer]{answer}"
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams1)
        #print("finish step1")
        # step 2: generate new answer
        prompt = f"Here is a format instruction, a user question, a candidate answer, and an analysis.\nPlease incorporate the analysis, and then generate a new answer which strictly follows the format instruction.\n"
        prompt += f"[Format Instruction]{instruction}\n[User Question]{query}\n[Candidate Answer]{answer}\n[Analysis]{res}"
        res2 = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams1)
        #print("finish step2")

        return res2


class Judger:
    def __init__(self,
                 name='openaichat/gpt-4-1106-preview',
                 s_id=0,
                 temp=0.1,

                 ):
        self.trial=s_id
        self.model=name
        self.temp=temp
        return

    def judge(self,
                query,
                #instruction,
                answer,
                ):
        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=self.trial,
        )
        prompt = f"[User Question]:{query}\n[Answer]:{answer}\n"
        prompt += 'Instruction: Review your previous answer and find problems with your answer. Finally, conclude with either ’[[correct]]’ if the above answer is correct or ’[[wrong]]’ if it is incorrect. Think step by step.\nVerdict:'
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams)

        if("[correct]" in res):
            return 1
        else:
            return 0


class Filter(object):
    def __init__(self,
                 name='openaichat/gpt-4-1106-preview',
                 s_id=0,
                 temp=0.1,
                 judge_type="multi_step",
                 ):
        self.trial=s_id
        self.model=name
        self.temp=temp
        self.judge_type = judge_type
        self.myjudger = Judger(name=name,s_id=s_id,temp=temp)
        if(judge_type=='multi_step'):
            self.myjudger = JudgerMultiSteps(name=name,s_id=s_id,temp=temp)

        if(judge_type=='single_step'):
            self.myjudger = Judger(name=name,s_id=s_id,temp=temp)

        if (judge_type=='ksp' or judge_type=='ksp_finegrained' ):
            self.myjudger = JudgerKSP(name=name,s_id=s_id,temp=temp)
        self.judge_type = judge_type
        self.max_workers=200
        return

    def filter(self,
                query,
                answers,
                ):
        #judgement = [self.myjudger.judge(query = query, answer =a) for a in answers]
        judgement = self.parallel_judgement(query=query,answers=answers)
        new_ans = []
        for idx, a in enumerate(answers):
            if(judgement[idx]>=0.5):
                new_ans.append(a)
        if(len(new_ans)==0):
            return answers
        if(self.judge_type=='ksp_finegrained'):
            value_1 = 0
            ans = new_ans[0]
            for output in new_ans: 
                solution, reasoning = parse_xml_to_dict(output)
                total_value = int(solution.get('TotalValue', -1))
                if(total_value>value_1):
                    ans = output
                    value_1 = total_value
            return [ans]
        return new_ans      

    def parallel_judgement(self, query, answers):
        # Function to be executed in parallel
        def task(answer):
            return self.myjudger.judge(query=query, answer=answer)

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Schedule the execution of each task
            futures = [executor.submit(task, answer) for answer in answers]
            # Wait for all futures to complete and extract results
            judgement = [future.result() for future in futures]

        return judgement


class JudgerMultiSteps:
    def __init__(self,
                 name='openaichat/gpt-4-1106-preview',
                 s_id=0,
                 temp=0.1,
                instruction='',
                step_k=3,
                 ):
        self.trial=s_id
        self.model=name
        self.temp=temp
        self.instruction = instruction
        self.step_k = step_k
        return

    def judge(self,
                query,
                #instruction,
                answer,
                ):
        
        answers_list = answer.split("\n")
        step_k = self.step_k
        likelihood = []
        #print("answer list is",answers_list)
        for idx, a in enumerate(answers_list):
            s1 = 0
            for j in range(step_k):
                context = '\n'.join(answers_list[0:idx-1])
                if(idx==0):
                    context = 'Initial State.'
                s1+=self.get_score_single(query,context,a,j)
            s1/=step_k
            likelihood.append(s1)
        #print("likelihood",likelihood)
        return numpy.average(likelihood)
    def get_score_single(self,query, context,a,k=0):
        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=k,
        )
        prompt = f"[Start of Question]\n{query}\n[End of Question]\n[Start of Previous Derivations]\n{context}\n[End of Previous Derivations]\n"
        prompt += f'[Instruction] Suppose the above derivations are correct. Following the last iteration, is it correct to assert that "{a}"? Your final answer should be [[yes]] or [[no]].\n[Analysis]\nLet us think step by step.'
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams)
        #print("RES::",res)
        if("[yes]" in res):
            return 1
        elif("[no]" in res):
            return 0
        else:
            return 0.5


class JudgerKSP:
    def __init__(self,
                 name='openaichat/gpt-4-1106-preview',
                 s_id=0,
                 temp=0.1,
                instruction='',
                step_k=3,
                 ):
        self.trial=s_id
        self.model=name
        self.temp=temp
        self.instruction = instruction
        self.step_k = step_k
        return

    def judge(self,
                query,
                #instruction,
                answer,
                ):
        #correct, reason = kspCheck(query, answer)

        try:
            correct, reason = kspCheck(query, answer)
            #print("correct, reason",correct, reason)
        except:
            correct = False
        return correct

    def get_score_single(self,query, context,a,k=0):
        genparams = GenerationParameter(max_tokens=3000, 
                                temperature=self.temp,
                                stop='\n\n\n\n\n\n',
                                date=20240206,
                                trial=k,
        )
        prompt = f"[Start of Question]\n{query}\n[End of Question]\n[Start of Previous Derivations]\n{context}\n[End of Previous Derivations]\n"
        prompt += f'[Instruction] Suppose the above derivations are correct. Following the last iteration, is it correct to assert that "{a}"? Your final answer should be [[yes]] or [[no]].\n[Analysis]\nLet us think step by step.'
        res = MyLLMEngine.getcompletion(query=prompt,service_name=self.model,genparams=genparams)
        #print("RES::",res)
        if("[yes]" in res):
            return 1
        elif("[no]" in res):
            return 0
        else:
            return 0.5
