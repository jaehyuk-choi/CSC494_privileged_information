import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from ucimlrepo import fetch_ucirepo
from openai import OpenAI
import json


def get_patient_descr_891():
    descr_template = "A {sex} patient aged {age} with BMI {bmi}, {bp} high blood pressure reported and "\
                    '{chol} high cholesterol reported. They have {CholCheck} had a cholesterol check in the last 5 years. ' \
                    'The patient has {smoke} and {alcohol}. '\
                    'They have {stroke} had a stroke, {cvd} coronary heart disease or a myocardial infarction and '\
                    'rate their general health as a {genhealth} on a scale of 1-5 where 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor. '\
                    'The patient says their physical health was not good for {PhysHlth} out of the last 30 days, '\
                    'and their mental health was not good for {MentHlth} out of the last 30 days. '\
                    'They report having {diffwalk} difficulty walking or climbing stairs. '\
                    'The patient reports {PhysActivity} having been physically active in the past 30 days, {veggies} consuming '\
                    'vegetables daily, and {fruits} consuming fruits daily. '\
                    'They {AnyHealthcare} have healthcare coverage, and {NoDocbcCost} report not seeing a doctor because '\
                    "of cost. Their highest education level attained is: {education}, and income is {income}."
    return descr_template

def get_field_mapping_891():
    mapping = {
        "sex": lambda x: 'male' if x['Sex'] == 1 else 'female',
        "age": lambda x: '18-24' if x['Age'] == 1 else '25-29' if x['Age'] == 2 else '30-34' if x['Age'] == 3 else \
                        '35-39' if x['Age'] == 4 else '40-44' if x['Age'] == 5 else '45-49' if x['Age'] == 6 else \
                        '50-54' if x['Age'] == 7 else '55-59' if x['Age'] == 8 else '60-64' if x['Age'] == 9 else \
                        '65-69' if x['Age'] == 10 else '70-74' if x['Age'] == 11 else '75-79' if x['Age'] == 12 else '80+',
        "bmi":'BMI',
        "bp": lambda x: '' if x['HighBP'] == 1 else 'no',
        "chol": lambda x: '' if x['HighChol'] == 1 else 'no',
        "smoke": lambda x: 'smoked (>5 packs in lifetime)' if x['Smoker'] == 1 else 'never smoked',
        "alcohol": lambda x: 'drinks alcohol heavily' if x['HvyAlcoholConsump'] == 1 else 'is not a heavy drinker',
        "CholCheck": lambda x: '' if x['CholCheck'] == 1 else 'not',
        "stroke": lambda x: '' if x['Stroke'] == 1 else 'never',
        "cvd": lambda x: 'have had' if x['HeartDiseaseorAttack'] == 1 else 'have never had',
        "genhealth": 'GenHlth',
        "PhysHlth": 'PhysHlth',
        "MentHlth": 'MentHlth',
        "diffwalk": lambda x: '' if x['DiffWalk'] == 1 else 'no',
        "PhysActivity": lambda x: '' if x['PhysActivity'] == 1 else 'not',
        "veggies": lambda x: '' if x['Veggies'] == 1 else 'not',
        "fruits": lambda x: '' if x['Fruits'] == 1 else 'not',
        "AnyHealthcare": lambda x: '' if x['AnyHealthcare'] == 1 else 'do not',
        "NoDocbcCost": lambda x: '' if x['NoDocbcCost'] == 1 else 'do not',
        "education": lambda x: 'no schooling' if x['Education'] == 1 else 'elementary school' if x['Education'] == 2 else \
                                'some high school' if x['Education'] == 3 else 'high school graduate' if x['Education'] == 4 else \
                                'some college' if x['Education'] == 5 else 'college graduate',
        "income": lambda x: 'less than $10,000' if x['Income'] == 1 else '$10,000 to $15,000' if x['Income'] == 2 else \
                            '$15,000 to $20,000' if x['Income'] == 3 else '$20,000 to $25,000' if x['Income'] == 4 else \
                            '$25,000 to $35,000' if x['Income'] == 5 else '$35,000 to $50,000' if x['Income'] == 6 else \
                            '$50,000 to $75,000' if x['Income'] == 7 else 'over $75,000'
    }
    return mapping  


def get_prompts_891(df):
    
    aux_patterns = ['has_diabetes','health_1_10','diabetes_risk_score','predict_hba1c','predict_bp','predict_cholesterol',
                    'exercise_freq', 'hi_sugar_freq','employment_status']
    prompts = {k:[] for k in aux_patterns}
    sys_prompt = {"role": "system","content": "You are a doctor evaluating patients who may or may not have diabetes. "\
                    "You will be given a detailed description of each patient and answer a question about them. The answers " \
                    "should only be numeric answers as specified in each question. Do not answer in any other way."}#\
                    # "If you are unable to provide an estimate for the answer, respond only with an empty string."}

    pattern_prompts = {'has_diabetes' : ' Does this patient have diabetes or pre-diabetes? Answer 1 for yes or 0 for no. Answer only with the number 0 or 1, do not answer with anything other than this number.',
                       'health_1_10' : ' On a scale of 1 to 10 how healthy is this patient. Answer only with a number from 1 to 10. Do not answer with anything other than this number.',
                       'diabetes_risk_score' : ' Give a risk score in percent of this patient developing diabetes is the diabetes. Answer only with a number from 0 to 100. Do not answer with anything other than this number.',
                       'predict_hba1c' : ' Give your best estimate of the HbA1C percentage of this patient. Answer only with the percentage. Do not answer with anything other than this number.',
                       'predict_bp' : ' Give your best estimate of the blood pressure of this patient. Answer only with the two numbers for systolic and diastolic blood pressure, separated by a forward slash. Do not answer with anything other than these two numbers.',
                       'predict_cholesterol' : ' Give your best estimate of the total cholesterol of this patient. Answer only with a single number. Do not answer with anything other than this number.',
                       'exercise_freq' : ' Give your best estimate of the number of times this patient exercises per week? Answer only with a single integer. Do not answer with anything other than this number.',
                       'hi_sugar_freq' : ' Give your best estimate of the number of times this patient consumes high sugar foods per week? Answer only with a single integer. Do not answer with anything other than this number.',
                       'employment_status' : ' What is the likely employment status of this patient? Answer 1 if they are currently employed or 0 if they are currently unemployed. Answer only 1 or 0, do not answer with anything else.'
                      }
    descr_template = get_patient_descr_891()    
    field_mapping = get_field_mapping_891()
    
    for _, row in df.iterrows():
        fill_prompt = {
            field: (func(row) if callable(func) else row[func]) for field, func in field_mapping.items()
        }
        for k,v in pattern_prompts.items():
            descr_prompt = {"role": "user", "content": descr_template.format(**fill_prompt)+v}
            prompts[k].append([sys_prompt,descr_prompt])
    return prompts

def get_pairwise_prompts_891(df):
    aux_patterns = ['similarity']
    prompts = {k:[] for k in aux_patterns}
    sys_prompt = {"role": "system","content": "You are a doctor evaluating patients who may or may not have diabetes. "\
                    "You will be given a detailed description of two patients and asked to compare them. The answers " \
                    "should only be numeric answers as specified in each question. Do not answer in any other way."}#\
                    # "If you are unable to provide an estimate for the answer, respond only with an empty string."}

    pattern_prompts = {'similarity' : ' On a scale of 0 to 10 how similar are these two patients, where 0 is completely dissimilar and 10 is completely similar. Answer only with a number from 0 to 10. Do not answer with anything other than this number.'
                      }
    descr_template = get_patient_descr_891()    
    field_mapping = get_field_mapping_891()
    
    pairs = {}
    # for each row in df, get 10 other random rows, format the template for both of them and connect together.
    for _, row in df.iterrows():
        # get 10 other random rows
        pairs[int(row['index'])] = []
        other_rows = df.sample(10)
        for _, other_row in other_rows.iterrows():
            pairs[int(row['index'])].append(int(other_row['index']))
            fill_prompt = {
                field: (func(row) if callable(func) else row[func]) for field, func in field_mapping.items()
            }
            fill_prompt_other = {
                field: (func(other_row) if callable(func) else other_row[func]) for field, func in field_mapping.items()
            }
            
            
            descr_prompt = {"role": "user", "content": "Patient 1: " + descr_template.format(**fill_prompt)+\
                            " Patient 2: " + descr_template.format(**fill_prompt_other)+pattern_prompts['similarity']}
            prompts['similarity'].append([sys_prompt,descr_prompt])
    return prompts, pairs  

def get_prompts(dataset_id,df):
    if dataset_id == 891:
        return get_prompts_891(df)
    
def get_pairwise_prompts(dataset_id,df):
    if dataset_id == 891:
        return get_pairwise_prompts_891(df)


def process_binary(key, res_dict):
    for i in range(len(res_dict[key])):
        try:
            val = float(res_dict[key][i])
            if val not in [0,1]:
                res_dict[key][i] = np.nan
            else:
                res_dict[key][i] = val
        except:
            res_dict[key][i] = np.nan
    return res_dict[key]

def process_count(key, res_dict):
    for i in range(len(res_dict[key])):
        try:
            val = int(res_dict[key][i])
            if val < 0:
                res_dict[key][i] = np.nan
            else :
                res_dict[key][i] = val
        except:
            res_dict[key][i] = np.nan
    return res_dict[key]

def process_range(key, res_dict, low=0,high=100):
    for i in range(len(res_dict[key])):
        try:
            val = float(res_dict[key][i])
            if val < low or val > high:
                res_dict[key][i] = np.nan
            else:
                res_dict[key][i] = val  # Keep valid values as floats
        except:
            res_dict[key][i] = np.nan
    return res_dict[key]

def post_process_pairwise_891(res_dict):
    res_dict['similarity'] = process_range('similarity', res_dict,0,10)
    return res_dict
    

def post_process_891(res_dict):

    res_dict['has_diabetes'] = process_binary('has_diabetes', res_dict)
    
    res_dict['health_1_10'] = process_range('health_1_10',res_dict,1,10)
    
    res_dict['diabetes_risk_score'] = process_range('diabetes_risk_score',res_dict,0,100)
    res_dict['predict_hba1c'] = process_range('predict_hba1c',res_dict,0,100)
    
    for i in range(len(res_dict['predict_bp'])):
        try:
            v = res_dict['predict_bp'][i].split('/')
            if len(v) != 2:
                v = [np.nan,np.nan]
            else:
                v = [float(v[0]),float(v[1])]
            res_dict['predict_bp'][i] = v
        except:
            v = [np.nan,np.nan]
            res_dict['predict_bp'][i] = v
    res_dict['systolic_bp'] = [v[0] for v in res_dict['predict_bp']]
    res_dict['diastolic_bp'] = [v[1] for v in res_dict['predict_bp']]
    del res_dict['predict_bp']
        
    for i in range(len(res_dict['predict_cholesterol'])):
        try:
            v = float(res_dict['predict_cholesterol'][i])
            if v < 0:
                res_dict['predict_cholesterol'][i] = np.nan
            else:
                res_dict['predict_cholesterol'][i] = v
        except:
            res_dict['predict_cholesterol'][i] = np.nan

    res_dict['exercise_freq'] = process_count('exercise_freq', res_dict)
    res_dict['hi_sugar_freq'] = process_count('hi_sugar_freq', res_dict)
    res_dict['employment_status'] = process_binary('employment_status', res_dict)
    
    return res_dict
    

def post_process(res_dict,dataset_id,pairwise=False):
    if dataset_id == 891:
        if pairwise:
            res=post_process_pairwise_891(res_dict)
        else:
            res=post_process_891(res_dict)
    res_df = pd.DataFrame(res)
    return res_df


def generate_prompts(df,promptfile,dataset_id,pairwise=False):
    if pairwise:
        prompts, pairs = get_pairwise_prompts(dataset_id,df)
        with open('pairs_70B.json','w') as f:
            print(pairs)
            json.dump(pairs,f,indent=4)
    else:    
        prompts = get_prompts(dataset_id,df)
    with open(promptfile,'w') as f:
        json.dump(prompts,f,indent=4)
        # for k,v in prompts.items():
        #     f.write(f'{k}\n')
        #     for p in v:
        #         f.write(f'{p}\n')
    return prompts

def run_inf(prompt_list):
    # The url is located in the .vLLM_model-variant_url file in the corresponding model directory.
    client = OpenAI(base_url="http://gpu042:8080/v1", api_key="EMPTY")

    # Update the model path accordingly
    solutions = {k:[] for k in prompt_list.keys()}
    for k,prompts in prompt_list.items():
        print("Getting results for prompt: ",k)
        for i,prompt in enumerate(prompts):
            if i % 500 == 0:
                print(f'Prompt {i} of {len(prompts)}',flush=True)
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-70B-Instruct",
                messages=prompt,
            )
            solutions[k].append(completion.choices[0].message.content)
    return solutions

    
    
def get_features(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features
    labels = dataset.data.targets
    df = pd.concat([features,labels],axis=1)
    return df


def main(promptfile='prompts_70B.json', datafile ='data.csv', resfile='augmented_data_70B.csv', pairwise=False):
    if not os.path.exists(promptfile):
        
        df = get_features(dataset_id=891)
        
        # Stratified sampling
        df_0 = df[df['Diabetes_binary'] == 0].sample(5000)
        df_1 = df[df['Diabetes_binary'] == 1].sample(5000)

        # Combine the two samples and shuffle
        df = pd.concat([df_0, df_1]).sample(frac=1).reset_index()
        df.to_csv(datafile,index=False)
            
        prompt_dict = generate_prompts(df,promptfile,dataset_id=891,pairwise=pairwise)

        res_df = pd.DataFrame()
        num_done = 0
        seen = 0
    
    else:   # read the prompts from the file
        print("Reading prompts from file")
        with open(promptfile,'r') as f:
            prompt_dict = json.load(f)
        # start the prompt list from where the previous run ended, based on resfile
        df = pd.read_csv(datafile)
        res_df = pd.read_csv(resfile)
        num_done = len(res_df)
        seen = len(res_df)
        if pairwise:
            seen = int(len(res_df)/10)
        remaining_prompts = {}
        print(f"Resuming from {num_done}")
        for k,v in prompt_dict.items():
            print(f"Prompt {k} has {len(v)-(num_done)} prompts remaining")
            remaining_prompts[k] = v[num_done:]
        prompt_dict = remaining_prompts
        
    
    # chunk the prompt list:
    n = 500
    chunk_start = 0
    for i in range(seen, len(df), n):
        print(i)
        if pairwise:
            prompt_list_chunk = {k:v[chunk_start:chunk_start+n*10] for k,v in prompt_dict.items()}
        else:     
            prompt_list_chunk = {k:v[chunk_start:chunk_start+n] for k,v in prompt_dict.items()}
        
        res_chunk = run_inf(prompt_list_chunk)
            
        res_df_chunk = post_process(res_chunk, dataset_id=891, pairwise=pairwise)
        if pairwise:
            # 10 rows in df correspond to one patient, so repeat the rows 10 times
            # generate a df that contains each row in df copied consecutive 10 times
            
            df_copied = df.iloc[seen:seen+n].reset_index(drop=True)
            
            df_copied = df_copied.loc[np.repeat(df_copied.index, 10)].reset_index(drop=True)
            aug_df = pd.concat([df_copied,res_df_chunk.reset_index(drop=True)],axis=1)
                        
        else:
            aug_df = pd.concat([df.iloc[seen:seen+n].reset_index(drop=True),res_df_chunk.reset_index(drop=True)],axis=1)
        
        print(aug_df.head())
        
        res_df = pd.concat([res_df,aug_df])
            
        res_df.to_csv(resfile,index=False)
        if pairwise:
            chunk_start += n*10
        else:
            chunk_start += n
        seen += n

def test():
    df = pd.read_csv('augmented_data_70B_new.csv')
    print(df['has_diabetes'].isna().sum())
    print((df['Diabetes_binary']==df['has_diabetes']).sum())
    print(f'TP: {((df["Diabetes_binary"]==1)&(df["has_diabetes"]==1)).sum()} TN: {((df["Diabetes_binary"]==0)&(df["has_diabetes"]==0)).sum()}')
    print(f'FP: {((df["Diabetes_binary"]==0)&(df["has_diabetes"]==1)).sum()} FN: {((df["Diabetes_binary"]==1)&(df["has_diabetes"]==0)).sum()}')
    
    df_old = pd.read_csv('augmented_data.csv')
    print(df_old['has_diabetes'].isna().sum())
    print((df_old['Diabetes_binary']==df_old['has_diabetes']).sum())
    print(f'TP: {((df_old["Diabetes_binary"]==1)&(df_old["has_diabetes"]==1)).sum()} TN: {((df_old["Diabetes_binary"]==0)&(df_old["has_diabetes"]==0)).sum()}')
    print(f'FP: {((df_old["Diabetes_binary"]==0)&(df_old["has_diabetes"]==1)).sum()} FN: {((df_old["Diabetes_binary"]==1)&(df_old["has_diabetes"]==0)).sum()}')
    
if __name__ == "__main__":
    test()
    # main(promptfile='prompts_70B_new.json', resfile='augmented_data_70B_new.csv',pairwise=False)
