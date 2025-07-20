
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# 모델 불러오기
@st.cache_resource
def load_model():
    clf = joblib.load("bert_logistic_model.pkl")
    sbert = SentenceTransformer("sbert_model")
    return clf, sbert

clf, sbert = load_model()

# 제목
st.title("🧠 AI 댓글 판별기 (BERT 기반)")
st.write("댓글을 입력하면 AI가 작성한 것인지 판단해줍니다.")

# 사용자 입력
user_input = st.text_area("💬 댓글을 입력하세요:", height=100)

# 예측 함수
def predict_ai_comment(comment):
    embed = sbert.encode([comment])
    pred = clf.predict(embed)[0]
    prob = clf.predict_proba(embed)[0][1]
    label = "🤖 AI가 작성한 댓글입니다." if pred == 1 else "🙋 사람이 작성한 댓글입니다."
    return label, prob

# 버튼 클릭 시 예측
if st.button("판별하기"):
    if user_input.strip() == "":
        st.warning("댓글을 입력해주세요.")
    else:
        result, confidence = predict_ai_comment(user_input)
        st.markdown(f"### {result}")
        st.markdown(f"📊 AI 확률: **{confidence*100:.2f}%**")
