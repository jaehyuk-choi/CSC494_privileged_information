from openai import OpenAI

# The url is located in the .vLLM_model-variant_url file in the corresponding model directory.
client = OpenAI(base_url="http://gpu043:8081/v1", api_key="EMPTY")

# Update the model path accordingly
completion = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a doctor evaluating patients who may or may not have diabetes."
        },
        {"role": "user", "content": "A male patient aged 60-64 with BMI 29 and high blood pressure says their physical health was not good for 10 out of the last 30 days. Does this patient have diabetes or pre-diabetes? Answer only 'yes' or 'no'."},
        {"role": "user", "content": "A male patient aged 24-29 with BMI 23 and income over $75000 rates their health as 4 on a scale of 1 to 5. Does this patient have diabetes or pre-diabetes? Answer only 'yes' or 'no'."},
    ],
)

print(completion.choices[0].message.content)

