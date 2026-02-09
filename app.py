import streamlit as st
import os
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

# 🔐 لایه امنیتی اصلی
# کد فقط به دنبال کلید در تنظیمات مخفی (Secrets) می‌گردد
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🔑 کلید API پیدا نشد! لطفا آن را در پنل Settings > Secrets در سایت Streamlit وارد کنید.")
    st.stop() # توقف اجرای برنامه در صورت نبود کلید

# باقی کد دقیقا همان منطق قبلی است...
st.set_page_config(page_title="پشتیبان هوشمند طاقچه", page_icon="📚")
st.markdown("<style>.stApp { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag():
    if not os.path.exists("data.csv"):
        return None, None, None, None
    
    df = pd.read_csv("data.csv", encoding="utf-8-sig")
    df["category"] = df["category"].fillna("نامشخص").astype(str).str.strip()
    df["answer"] = df["answer"].fillna("پاسخی ثبت نشده است").astype(str).str.strip()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    embeddings = OpenAIEmbeddings()
    
    categories = df["category"].unique().tolist()
    categories_str = "\n".join([f"- {c}" for c in categories])
    
    vectorstores = {}
    for cat in categories:
        cat_df = df[df["category"] == cat]
        answers = cat_df["answer"].tolist()
        if answers:
            vectorstores[cat] = FAISS.from_texts(texts=answers, embedding=embeddings)
            
    return llm, vectorstores, categories, categories_str

llm, vectorstores, categories, categories_str = initialize_rag()

# ... (ادامه توابع و رابط کاربری مشابه قبل)