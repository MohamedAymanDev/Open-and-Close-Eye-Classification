import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import pandas as pd

from PIL import Image


# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="RetinaSense AI",
    page_icon="🩺",
    layout="wide"
)



# ==============================
# Custom CSS
# ==============================

st.markdown("""
<style>


.stApp {

background:
linear-gradient(
135deg,
#020617,
#0f172a,
#1e293b
);

color:white;

}


/* Main Title */

h1 {

text-align:center;

font-size:60px;

font-weight:900;

color:#38bdf8;

text-shadow:
0px 0px 25px rgba(56,189,248,0.5);

}



h2,h3 {

color:white;

}



/* Result Card */

.card {

background:
rgba(255,255,255,0.08);


padding:35px;


border-radius:30px;


border:
1px solid rgba(255,255,255,0.15);


box-shadow:

0px 0px 35px rgba(56,189,248,0.3);


text-align:center;


}



/* Result Text */

.result {


font-size:50px;


font-weight:900;


color:#38bdf8;


}



/* Information Cards */

.info-card {


background:
rgba(255,255,255,0.06);


padding:30px;


border-radius:25px;


text-align:center;


border:

1px solid rgba(255,255,255,0.15);


transition:

all 0.35s ease;


cursor:pointer;


}



.info-card:hover {


transform:

translateY(-12px) scale(1.05);


background:

rgba(56,189,248,0.15);


box-shadow:

0px 15px 40px rgba(56,189,248,0.35);


border:

1px solid #38bdf8;


}



.info-icon {


font-size:45px;


margin-bottom:10px;


}



.info-title {


font-size:20px;


font-weight:bold;


color:#38bdf8;


}



.info-value {


font-size:18px;


color:white;


}



/* Analyze Button */

.stButton button {


background:

linear-gradient(
135deg,
#06b6d4,
#2563eb
);


color:white;


border:none;


border-radius:30px;


height:60px;


width:260px;


font-size:20px;


font-weight:bold;


transition:0.3s;


}



.stButton button:hover {


transform:

scale(1.08);


box-shadow:

0px 0px 35px #38bdf8;


background:

linear-gradient(
135deg,
#2563eb,
#06b6d4
);


}



</style>

""",
unsafe_allow_html=True
)




# ==============================
# Paths
# ==============================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "eye_classifier.keras"
)


ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)




# ==============================
# Load Model
# ==============================


@st.cache_resource
def load_model():


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    with open(
        ENCODER_PATH,
        "rb"
    ) as f:

        encoder = pickle.load(f)


    return model, encoder



model, LE = load_model()




# ==============================
# Header
# ==============================


st.markdown(
"""

<h1>
🩺 RetinaSense AI
</h1>


<p style="
text-align:center;
font-size:22px;
color:#cbd5e1;
">

Advanced Deep Learning System For Eye Disease Classification

</p>


""",

unsafe_allow_html=True
)



st.write("")




# ==============================
# Model Cards
# ==============================


col1,col2,col3,col4 = st.columns(4)



cards = [

("🧠","AI Model","CNN"),

("⚡","Engine","TensorFlow"),

("📐","Resolution","224 × 224"),

("🎯","Mission","Eye Classification")

]



for col,card in zip(
    [col1,col2,col3,col4],
    cards
):

    with col:

        st.markdown(
        f"""

        <div class="info-card">


        <div class="info-icon">

        {card[0]}

        </div>


        <div class="info-title">

        {card[1]}

        </div>


        <br>


        <div class="info-value">

        {card[2]}

        </div>


        </div>


        """,

        unsafe_allow_html=True
        )



st.write("")
st.write("")




# ==============================
# Upload Image
# ==============================


image_file = st.file_uploader(

    "📤 Upload Eye Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)





# ==============================
# Preprocessing
# ==============================


def preprocess(image):


    img=np.array(image)



    if len(img.shape)==2:

        gray=img


    elif img.shape[2]==1:

        gray=img[:,:,0]


    else:

        gray=cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )



    gray=cv2.resize(
        gray,
        (224,224)
    )


    gray=gray/255.0



    gray=np.expand_dims(
        gray,
        axis=-1
    )


    gray=np.expand_dims(
        gray,
        axis=0
    )


    return gray





# ==============================
# Prediction
# ==============================


if image_file:


    image=Image.open(
        image_file
    )


    st.markdown(
        "## 🖼 Image Preview"
    )


    st.image(
        image,
        width=350
    )



    if st.button(
        "🚀 Start AI Diagnosis"
    ):


        with st.spinner(
            "AI is analyzing the eye..."
        ):


            processed=preprocess(
                image
            )


            prediction=model.predict(
                processed
            )




        # Binary

        if prediction.shape[1]==1:


            prob=float(
                prediction[0][0]
            )


            index=1 if prob>=0.5 else 0


            label=LE.inverse_transform(
                [index]
            )[0]


            confidence=max(
                prob,
                1-prob
            )


            probs=pd.DataFrame(

                {

                "Probability":

                [
                    1-prob,
                    prob
                ]

                },

                index=LE.classes_

            )




        # Multi Class

        else:


            index=np.argmax(
                prediction
            )


            label=LE.inverse_transform(
                [index]
            )[0]


            confidence=np.max(
                prediction
            )


            probs=pd.DataFrame(

                prediction[0],

                index=LE.classes_,

                columns=[
                    "Probability"
                ]

            )





        # Result


        st.write("")



        st.markdown(
        f"""

        <div class="card">


        <h2>
        🧬 AI Diagnosis Result
        </h2>


        <div class="result">

        {label}

        </div>


        <h3>

        Confidence:

        {confidence*100:.2f}%

        </h3>


        </div>


        """,

        unsafe_allow_html=True

        )



        st.write("")



        # Chart


        st.markdown(
            "## 📊 Prediction Probability"
        )


        probs["Probability"] *= 100


        st.bar_chart(
            probs
        )


        st.success(
            "Diagnosis Completed Successfully ✅"
        )





# ==============================
# Footer
# ==============================


st.markdown(
"""

<br><br>

<hr>


<center style="
color:#94a3b8;
font-size:18px;
">

Powered by TensorFlow | CNN | Streamlit

<br>

Developed by Mohamed Ayman

</center>


""",

unsafe_allow_html=True
)