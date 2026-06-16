import os
import shutil
import streamlit as st
from typing import List
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 在线通义Embedding
from langchain_community.embeddings import DashScopeEmbeddings
# 混合检索 BM25 + 多路融合
# v1.x 混合检索正确导入
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="PDF混合检索RAG问答系统", layout="wide")
st.title("📚 PDF文档智能问答（混合检索+溯源引用）")

# ===================== 加载密钥（本地secrets/云端网页配置双兼容） =====================
def get_api_key():
    # 云端streamlit secrets优先级最高
    if "LLM_API_KEY" in st.secrets:
        return st.secrets["LLM_API_KEY"]
    # 本地读取.env
    load_dotenv()
    return os.getenv("LLM_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("未读取到LLM_API_KEY，请检查secrets或.env配置！")
    st.stop()

# ===================== 初始化在线Embedding =====================
@st.cache_resource
def init_embedding():
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=api_key
    )
    return embeddings

embedding = init_embedding()

# ===================== 初始化LLM =====================
@st.cache_resource
def init_llm():
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.01
    )
    return llm

llm = init_llm()

# ===================== 全局临时存储路径 =====================
UPLOAD_FOLDER = "./upload_pdf"
CHROMA_PATH = "./chroma_streamlit_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===================== PDF加载与分块 =====================
def load_pdf_and_split(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，"]
    )
    chunks = splitter.split_documents(raw_docs)
    return chunks

# ===================== 构建向量库 =====================
def build_vector_store(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
        collection_name="streamlit_rag"
    )
    return db

# ===================== 混合检索器：向量+BM25 =====================
def get_hybrid_retriever(all_chunks, top_k=10):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
        collection_name="streamlit_rag"
    )
    dense_retriever = db.as_retriever(search_kwargs={"k": top_k})
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = top_k
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

# ===================== Prompt与上下文格式化 =====================
trace_prompt = PromptTemplate.from_template("""
你是专业文档问答助手，仅允许使用参考资料内容作答，严禁编造信息。
回答结束必须单独一段输出【引用来源】，格式：
[序号] 文档文件名 | 页码 | 原文摘要（100字内）

参考资料：
{context}
用户问题：{question}
""")

def format_context(docs):
    ctx = ""
    for idx, doc in enumerate(docs):
        ctx += f"片段{idx+1}：{doc.page_content}\n\n"
    return ctx

# ===================== 侧边栏：PDF上传区域 =====================
with st.sidebar:
    st.header("📤 上传PDF文档")
    uploaded_file = st.file_uploader("选择PDF文件", type=["pdf"])
    build_flag = False
    chunk_list = None

    if uploaded_file is not None:
        # 保存临时PDF到本地
        temp_pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"已上传：{uploaded_file.name}")

        if st.button("构建知识库向量库"):
            with st.spinner("正在分块、调用在线Embedding生成向量..."):
                chunk_list = load_pdf_and_split(temp_pdf_path)
                build_vector_store(chunk_list)
                st.success(f"知识库构建完成，总分块：{len(chunk_list)}")
                st.session_state["chunks"] = chunk_list
                build_flag = True

# ===================== 主页面问答区域 =====================
st.subheader("💬 文档问答")
question = st.text_input("输入你的问题：", placeholder="例如：自动播放是否静音，调用的方法名是什么")
submit_btn = st.button("开始问答")

if submit_btn and question:
    if "chunks" not in st.session_state:
        st.warning("请先在侧边栏上传PDF并构建知识库！")
    else:
        with st.spinner("混合检索中..."):
            retriever = get_hybrid_retriever(st.session_state["chunks"], top_k=10)
            retrieve_docs = retriever.invoke(question)
            context_str = format_context(retrieve_docs)

            # 构建问答链并生成答案
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | trace_prompt
                | llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke({"context": context_str, "question": question})

        # 输出AI回答
        st.markdown("### 🤖 AI回答")
        st.write(answer)

        # 输出溯源详情
        st.markdown("### 📑 检索来源溯源详情")
        for idx, doc in enumerate(retrieve_docs):
            src = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "无页码")
            brief = doc.page_content[:120] + "..."
            st.markdown(f"""
            **片段{idx+1}**
            文件：{src} | 页码：{page}
            摘要：{brief}
            """)