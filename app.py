import os
import streamlit as st
from typing import List
import chromadb

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
from langchain.retrievers import EnsembleRetriever
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

# ===================== 全局临时存储路径（仅存上传PDF，向量全程内存） =====================
UPLOAD_FOLDER = "./upload_pdf"
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

# ===================== 构建向量库（方案2：内存客户端，自动清旧集合） =====================
def build_vector_store(chunks):
    # 强制内存客户端，不读写磁盘
    client = chromadb.Client()
    coll_name = "streamlit_rag"
    # 存在同名集合先删除，避免重复写入rust报错
    try:
        client.delete_collection(name=coll_name)
    except Exception:
        pass
    # 新建内存向量库
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=coll_name,
        client=client
    )
    return db

# ===================== 混合检索器：复用已构建向量库，不再重复建库 =====================
def get_hybrid_retriever(vector_db, all_chunks, top_k=10):
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = top_k
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

# ===================== Query 扩展：多路同义改写 =====================
query_expand_prompt = PromptTemplate.from_template(
    "你是一个查询改写助手。请将用户的问题改写为3个不同角度的同义问题，"
    "每行一个，不要编号，不要解释，只要改写后的问题。\n"
    "用户问题：{question}"
)

def expand_queries(question: str, llm_instance) -> List[str]:
    """用 LLM 将单条 query 扩展为多条同义 query（含原始 query）"""
    prompt_text = query_expand_prompt.format(question=question)
    response = llm_instance.invoke(prompt_text)
    expanded = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    # 保留原始 query + 扩展 query，去重
    all_queries = [question] + expanded
    seen = set()
    unique = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique

def multi_query_retrieve(retriever, queries: List[str], top_k: int = 10):
    """多路检索 + 按内容去重合并"""
    seen_contents = set()
    merged_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            content_key = doc.page_content[:200]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged_docs.append(doc)
            if len(merged_docs) >= top_k * 2:
                break
        if len(merged_docs) >= top_k * 2:
            break
    return merged_docs[:top_k * 2]

# ===================== Prompt与上下文格式化 =====================
trace_prompt = PromptTemplate.from_template("""
你是专业文档问答助手，仅允许使用参考资料内容作答，严禁编造信息。
请按以下思维链步骤作答：

【分析】分析用户问题的核心意图和关键信息点。
【检索】从参考资料中定位与问题最相关的内容，列出关键片段。
【推理】基于检索到的内容，逐步推理得出答案。
【结论】用简洁清晰的语言给出最终答案。

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

    if uploaded_file is not None:
        # 保存临时PDF到本地
        temp_pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        # 仅在文件未处理过时执行保存与构建
        if st.session_state.get("last_uploaded_file") != uploaded_file.name:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            with st.spinner("正在分块、调用在线Embedding生成向量..."):
                chunk_list = load_pdf_and_split(temp_pdf_path)
                vector_db = build_vector_store(chunk_list)
                st.session_state["chunks"] = chunk_list
                st.session_state["vector_db"] = vector_db
                st.session_state["last_uploaded_file"] = uploaded_file.name
            st.success(f"知识库构建完成，总分块：{len(chunk_list)}")
        else:
            st.info(f"知识库已就绪：{uploaded_file.name}（总分块：{len(st.session_state.get('chunks', []))}）")

# ===================== 主页面问答区域 =====================
st.subheader("💬 文档问答")
question = st.text_input("输入你的问题：", placeholder="例如：自动播放是否静音，调用的方法名是什么")
submit_btn = st.button("开始问答")

if submit_btn and question:
    # 双重校验缓存：必须同时存在分块和向量库实例
    if "chunks" not in st.session_state or "vector_db" not in st.session_state:
        st.warning("请先在侧边栏上传PDF并构建知识库！")
    else:
        with st.spinner("正在扩展查询并混合检索..."):
            # 1. Query 扩展
            queries = expand_queries(question, llm)
            st.markdown("**🔍 查询扩展结果：**")
            st.write(" → ".join(queries))

            # 2. 构建混合检索器
            retriever = get_hybrid_retriever(
                vector_db=st.session_state["vector_db"],
                all_chunks=st.session_state["chunks"],
                top_k=10
            )

            # 3. 多路检索 + 去重合并
            retrieve_docs = multi_query_retrieve(retriever, queries, top_k=10)
            context_str = format_context(retrieve_docs)

            # 4. 流式输出
            st.markdown("### 🤖 AI回答")
            answer_placeholder = st.empty()
            full_answer = ""
            for chunk in llm.stream(trace_prompt.format(context=context_str, question=question)):
                full_answer += chunk.content
                answer_placeholder.markdown(full_answer)
            st.session_state["last_answer"] = full_answer