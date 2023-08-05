# 🎸 Image2Music
**사진에 어울리는 노래가 듣고 싶을때**

image2music 지금 보고 있는 풍경에 맞는 노래를 찾고 싶을 때를 위한 모델입니다.

이미지를 넣으면 어울리는 노래(아티스트, 제목)와 해당 노래가 속한 유튜브 플레이리스트를 보여줍니다.

SNS에 사진과 어울리는 노래를 찾아 공유하고 싶을 때, 풍경 분위기에 맞는 플레이리스트를 추천받고 싶을 때 이용할 수 있습니다.

## Composition
파일 구성은 크게 두 개의 디렉토리 구조로 구분되어 있습니다.

- 모델 파트
    - 학습, 추론, 배포 등의 코드가 포함되어있습니다.
- 크롤링 파트
    - 학습을 위한 유튜브 playlist 데이터를 수집하는 코드가 포함되어있습니다.

## Installation
Open a terminal and run:
```
$ conda create -n image2music python=3.9
$ conda activate image2music

$ pip3 install -r requirements.txt
```
## Run
Open a terminal and run:

### just deploy and try
```
$ streamlit run image_classification/deploy.py
```

### train & inference

- crawl playlist data
    - `youtube_crawl/assets/youtube_tag.txt` 파일에 유튜버 이름 태그를 넣어주세요
    - 단, 유튜버들의 플레이리스트는 다음처럼 더보기란에 음악 속성이 있어야 합니다.

    ```
    # youtube crawl
    $ python3 youtube_crawl/run.py

    # concat data to get train & valid data
    $ python3 youtube_crawl/merge_csvs.py
    ```

- train & inference

  ```
  # train
  $ python3 image_classification/run.sh

  # inference
  $ python3 image_classification/inference.py --image_url https://health.chosun.com/site/data/img_dir/2023/05/31/2023053102582_0.jpg
  ```
## Example
<img width="750" alt="스크린샷 2023-08-03 오후 10 49 53" src="https://github.com/vail131/image2music/assets/52443401/c8da8f07-ec69-4658-aba6-a98c6dc7c6c4">

## Limitation of this project
모델은 youtube playlist의 썸네일을 보고 해당 playlist의 라벨로 분류하도록 학습됐습니다.
주어진 입력 이미지와 유사한 썸네일을 가진 playlist를 반환할 것이므로 이 프로젝트는 플레이리스트를 만든 유튜버의 안목이 좋다는 걸 가정합니다. 😅
왜냐하면 이미지와 플레이리스트가 어울리는지 판별하는 것은 감성적인 영역으로 정량적 평가가 어렵기 때문입니다.
따라서 이미지와 어울리지 않는 플레이리스트가 나올 수 있고 해당 곡과 유사하지 않은 다른 곡들로 채워진 플레이리스트일 수 있음을 이해해 주시길 바랍니다.
