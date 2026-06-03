#load packages




from transformers import pipeline
import pandas as pd

#text classification

text = """Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package,I discovered to my horror that I had been sent ana action figure of Megatron\
 instead! As a lifelong enemy of the Deceptions, I hope you can understand my dilemma. To resolve the issue, I demand an exchnage of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

classifier = pipeline("text-classification")
outputs = classifier(text)
print(pd.DataFrame(outputs))


# Named entity recognition
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
print(pd.DataFrame(outputs))

#Question answering
#reader = pipeline("question-answering")
#question = "What does the customer want?"
#outputs = reader(question=question, context=text)
#print(pd.DataFrame(outputs))

#Summarization
summarizer = pipeline("summarization")
outputs = summarizer(text,max_length =45, clean_up_tokenization_spaces = True)
print(outputs[0]['summary_text'])
