import pandas as pd
from pypdf import PdfReader
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from huggingface_hub import login

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = {'feature': ""}
    stop_words = set(stopwords.words("english"))
    for sent in sentences:
        if any(criteria in sent for criteria in ['skills', 'education']):
            words = word_tokenize(sent)
            words = [word for word in words if word not in stop_words]
            tagged_words = pos_tag(words)
            filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
            features['feature'] += " ".join(filtered_words)
    return features

def process_resume_data(df):
    id = df['ID']
    category = df['Category']
    text = extract_text_from_pdf(f"/home/ham/mnt/nas/project/kupu/dataset/resume_data/data/data/{category}/{id}.pdf")
    features = preprocess_text(text)
    df['Feature'] = features['feature']
    return df

def get_embeddings(text, tokenizer, model):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True).to("cuda")
    # outputs = AutoModel.from_pretrained(model_name)
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().to("cpu").numpy()
    return embeddings

def print_top_matching_resumes(result_group):
    for i in trange(15):
        print("\nJob ID:", i)
        print("Cosine Similarity | Domain Resume | Domain Description")
        print(result_group.get_group(i)[['similarity', 'domainResume', 'domainDesc']])

def main():
    # access_token_read = "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"
    # login(token = access_token_read)

    # resume_data = pd.read_csv("/home/ham/mnt/nas/project/kupu/dataset/resume_data/Resume/Resume.csv")
    # resume_data = resume_data.drop(["Resume_html"], axis=1)
    # resume_data = resume_data.apply(process_resume_data, axis=1)
    # resume_data = resume_data.drop(columns=['Resume_str'])
    # resume_data.to_csv("/home/ham/mnt/nas/project/kupu/dataset/resume_job_desc.csv", index=False)
    resume_data = pd.read_csv("/home/ham/mnt/nas/project/kupu/dataset/resume_job_desc.csv")

    job_description = pd.read_csv("/home/ham/mnt/nas/project/kupu/dataset/resume_job_desc/training_data.csv")
    job_description = job_description[["job_description", "position_title"]][:15]
    job_description['Features'] = job_description['job_description'].apply(lambda x : preprocess_text(x)['feature'])

    device = "cuda" 
    model_name = "bert-base-uncased"
    # model_name = "distilbert-base-uncased"
    # model_name = "roberta-base"
    # model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    job_desc_embeddings = np.array([get_embeddings(desc, tokenizer, model) for desc in tqdm(job_description['Features'])]).squeeze()
    resume_embeddings = np.array([get_embeddings(text, tokenizer, model) for text in tqdm(resume_data['Feature'])]).squeeze()

    result_df = pd.DataFrame(columns=['jobId', 'resumeId', 'similarity', 'domainResume', 'domainDesc'])

    for i, job_desc_emb in tqdm(enumerate(job_desc_embeddings)):
        similarities = cosine_similarity([job_desc_emb], resume_embeddings)
        top_k_indices = np.argsort(similarities[0])[::-1][:10]
        for j in top_k_indices:
            result_df.loc[i+j] = [i, resume_data['ID'].iloc[j], similarities[0][j], resume_data['Category'].iloc[j], job_description['position_title'].iloc[i]]

    
    result_df = result_df.sort_values(by='similarity', ascending=False)
    result_group = result_df.groupby("jobId")
    print_top_matching_resumes(result_group)

if __name__ == "__main__":
    main()