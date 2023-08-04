import streamlit as st
from inference import ImageClassificationModel, inference, pd, pred2playlist, torch
from utils import show_image

st.set_page_config(page_title="Image2Playlist", page_icon="🎸")


@st.cache_resource
def get_environment():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_df = pd.read_csv("image_classification/data/train_data.csv")
    valid_df = pd.read_csv("image_classification/data/valid_data.csv")
    total_playlists_df = pd.concat([train_df, valid_df]).reset_index(drop=True)
    num_labels = len(total_playlists_df["vid_label"].unique())
    model = ImageClassificationModel(num_labels=num_labels)
    model.load_state_dict(torch.load("image_classification/ckpt/5_step13020.pt"))
    model.eval().to(device)

    return {
        "device": device,
        "model": model,
        "playlists_df": total_playlists_df,
    }


# cache environment
env_dict = get_environment()
device = env_dict["device"]
model = env_dict["model"]
playlists_df = env_dict["playlists_df"]

# interface
image_url = st.text_input("이미지 url을 입력하세요. 어울리는 노래를 추천해드릴게요~")
submit_button = st.button(label="Submit")

# 버튼이 눌렸을 때 동작
if submit_button:
    # 유저가 입력한 이미지를 보여준다
    st.markdown("""---""")
    image = show_image(image_url=image_url)
    input_container = st.container()
    input_container.caption("input image")
    input_container.caption("아래 이미지를 분석중이에요 ...")
    st.image(image)

    prediction = inference(model=model, image_url=image_url)
    song = pred2playlist(prediction=prediction, playlists=playlists_df)
    artist, title, video_id = song[["artist", "title", "video_id"]].values[0]
    st.markdown("""---""")
    output_container = st.container()
    output_container.caption("Result")

    song_rec_string = f"*{artist}* 의 노래 *{title}* 을 추천드립니다."
    output_container.markdown(song_rec_string)

    output_container.write("이 플레이리스트에 포함되어 있네요! 함께 수록된 노래들도 들어보시는건 어떨까요?")
    output_container.video(
        f"https://www.youtube.com/watch?v={video_id}", format="video/mp4"
    )
