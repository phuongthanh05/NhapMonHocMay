import streamlit as st
import pandas as pd
import re, string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# 1. Hàm tiền xử lý văn bản
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================
# 2. Streamlit GUI
# ============================================
st.title("📩 Spam SMS Detection - Naive Bayes (TF-IDF + ngram)")

menu = ["Huấn luyện mô hình", "Dự đoán tin nhắn", "Tải file kết quả"]
choice = st.sidebar.selectbox("Chọn chức năng", menu)

# ============================================
# 3. Huấn luyện mô hình
# ============================================
if choice == "Huấn luyện mô hình":
    st.subheader("🚀 Huấn luyện mô hình Naive Bayes")

    train_file = st.file_uploader("Upload file train.csv", type=["csv"])
    test_file = st.file_uploader("Upload file test.csv", type=["csv"])

    if train_file is not None and test_file is not None:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Làm sạch dữ liệu
        train_df["clean_text"] = train_df["sms"].apply(clean_text)
        test_df["clean_text"] = test_df["sms"].apply(clean_text)

        # TF-IDF + ngram
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        X_tfidf = tfidf_vectorizer.fit_transform(train_df["clean_text"])
        y = train_df["label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        nb_model = MultinomialNB(alpha=0.5)
        nb_model.fit(X_train, y_train)

        # Đánh giá
        y_pred_val = nb_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        st.success(f"🎯 Accuracy trên tập validation: {acc:.4f}")

        st.text("Báo cáo phân loại:")
        st.text(classification_report(y_val, y_pred_val))

        # Lưu model + vectorizer
        joblib.dump(nb_model, "naive_bayes_model.pkl")
        joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
        st.info("✅ Đã lưu mô hình thành công!")

# ============================================
# 4. Dự đoán tin nhắn mới
# ============================================
elif choice == "Dự đoán tin nhắn":
    st.subheader("🔎 Kiểm tra một tin nhắn mới")

    sms_input = st.text_area("Nhập nội dung tin nhắn tại đây:")

    if st.button("Dự đoán"):
        try:
            model = joblib.load("naive_bayes_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")

            clean_sms = clean_text(sms_input)
            X_input = vectorizer.transform([clean_sms])
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]

            ham_prob = proba[0] * 100
            spam_prob = proba[1] * 100

            if pred == 1:
                st.error(f"🚨 Đây là **SPAM**!\n\n"
                         f"📊 Xác suất: {spam_prob:.2f}% Spam | {ham_prob:.2f}% Ham")
            else:
                st.success(f"✅ Đây là **HAM (tin nhắn bình thường)**.\n\n"
                           f"📊 Xác suất: {ham_prob:.2f}% Ham | {spam_prob:.2f}% Spam")

        except:
            st.warning("⚠️ Bạn cần huấn luyện mô hình trước!")

# ============================================
# 5. Xuất file submission
# ============================================
elif choice == "Tải file kết quả":
    st.subheader("📂 Xuất file submission.csv")

    test_file = st.file_uploader("Upload file test.csv để dự đoán", type=["csv"])

    if test_file is not None:
        test_df = pd.read_csv(test_file)

        try:
            model = joblib.load("naive_bayes_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")

            test_df["clean_text"] = test_df["sms"].apply(clean_text)
            X_test_tfidf = vectorizer.transform(test_df["clean_text"])
            test_pred = model.predict(X_test_tfidf)

            submission = pd.DataFrame({
                "id": test_df["id"],
                "label": test_pred
            })

            submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")
            st.success("✅ Đã tạo file submission.csv")

            st.download_button(
                label="📥 Tải về submission.csv",
                data=submission.to_csv(index=False).encode("utf-8-sig"),
                file_name="submission.csv",
                mime="text/csv"
            )

        except:
            st.warning("⚠️ Bạn cần huấn luyện mô hình trước!")
