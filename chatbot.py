import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

load_dotenv()

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    return st.success('파일 업로드 성공!')

def generate_gpt_response(prompt):
    """OpenAI API 호출하여 GPT 응답 생성"""
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'),)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 모델 이름
            messages=[
                {"role": "system", "content": (
                    "당신은 운동 관련 도우미입니다. 사용자가 운동과 관련된 질문을 할 경우에만 대답하세요. "
                    "운동과 관련 없는 질문이 들어오면 정중히 거절 메시지를 반환하세요. "
                    "예: '죄송합니다, 저는 운동 관련 질문에만 답변할 수 있습니다.'")},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def main():
    #Streamlit 앱 제목
    st.title("🩶Gym equipment Chatbot")

    # 모델 불러오기
    model_path = "model_xtn.keras"
    if not os.path.exists(model_path):
        st.error("모델 파일이 없습니다. 경로를 확인하세요.")
        st.stop()
    model = load_model(model_path)
    CLASS_NAMES = ['Bench Press', 'Dumbbells', 'Elliptical Machine', 'Kettlebells',
              'Lat Pulldowns', 'Leg Curls', 'Leg Press',  'Recumbent Bike', 'Smith Machine']

    #이미지 업로드
    img_file = st.file_uploader('이미지를 업로드하세요.', type=['png', 'jpg', 'jpeg'])
    submit = st.button("업로드")

    if(submit):
        if img_file is not None:
            #이미지 로드 및 전처리
            img = load_img(img_file, target_size=(299, 299))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            #예측 수행
            y_pred = model.predict(img_array)
            st.write("예측 확률 분포:", y_pred)
            result = CLASS_NAMES[np.argmax(y_pred, axis=1)[0]]

            #결과 출력
            st.image(img_file, caption=f"Predicted: {result}", use_container_width=True)

            #GPT-4o API로 설명 생성
            prompt = f"{result}란 무엇이며 피트니스에서의 주요 용도는 무엇인가요? 한국어로 자세히 설명해주세요."
            with st.spinner("정보 불러오는 중..."):
                explanation = generate_gpt_response(prompt)
                st.write(explanation)
        else:
            st.warning("파일을 업로드하세요.")

    #####
    #대화#
    #####
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant", "content":"무엇을 도와드릴까요?"}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    #사용자 질문 저장 및 출력
    if prompt := st.chat_input("질문을 입력하세요."):
        st.session_state.messages.append({"role":"user",  "content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # OpenAI API 호출
        with st.spinner("응답 생성 중..."):
            assistant_response = generate_gpt_response(prompt)

        # 대화 기록 업데이트
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

if __name__ == "__main__":
    main()