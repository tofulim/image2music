# 🎸 Image2Playlist

---

**사진에 어울리는 노래가 듣고 싶을때**

image2playlist는 내가 지금 보고 있는 풍경에 맞는 노래를 찾고 싶을 때를 위한 모델입니다.

이미지를 넣으면 어울리는 노래(아티스트, 제목)와 해당 노래가 속한 유튜브 플레이리스트를 보여줍니다.

SNS에 사진과 함께 어울리는 노래를 찾고싶을 때

풍경 분위기에 맞는 플레이리스트를 추천받고 싶을 때 이용할 수 있습니다.

## Composition

---

파일 구성은 크게 두 개의 디렉토리 구조로 구분되어 있습니다.

- 모델 파트
    - 학습, 추론, 배포 등의 코드가 포함되어있습니다.
- 크롤링 파트
    - 유튜브 playlist를 가져와 모델을 학습하기 위한 데이터를 수집하는 코드가 포함되어있습니다.

## Installation

---

Open a terminal and run:

$ pip3 install -r requirements.txt

$

## Example
<img width="750" alt="스크린샷 2023-08-03 오후 10 49 53" src="https://github.com/vail131/image2music/assets/52443401/c8da8f07-ec69-4658-aba6-a98c6dc7c6c4">
