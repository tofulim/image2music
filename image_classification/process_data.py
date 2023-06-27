import argparse

import pandas as pd


def split_dataset(result_path: str, save_path: str):
    """
    playlist csv를 train, valid로 나눈다.
    유튜브 영상 1개(10개의 노래)를 {플레이리스트: 이미지} 로 매핑하여 vid_label을 부여하고 분리한다.

    Args:
        result_path (str): 통합 csv 파일이 들어있는 디렉토리
        save_path (str): 분리한 train/valid csv 파일을 저장할 위치

    Returns:
        None

    """
    total_data = pd.read_csv(result_path)

    # vid label column 추가
    vid2label = {
        vid: label for label, vid in enumerate(total_data["video_id"].unique())
    }

    total_data["vid_label"] = total_data["video_id"].apply(lambda x: vid2label[x])

    # vid label 기준으로 valid set 분리
    valid_df = pd.DataFrame()
    for vid_label in total_data["vid_label"].unique():
        temp_df = total_data.query("vid_label==@vid_label").sample(
            n=1
        )  # 그룹별 데이터 추출 및 2개 비복원 추출
        valid_df = pd.concat([valid_df, temp_df])  # 데이터 추가

    valid_df = valid_df.reset_index(drop=True)  # 인덱스 초기화

    # total에서 valid를 빼서 train set 만들기
    only_train_index_list = set(total_data.index) - set(valid_df.index)

    remained_train_data = total_data.iloc[list(only_train_index_list)]

    assert len(valid_df) + len(remained_train_data) == len(
        total_data
    ), "데이터 분리에 문제가 있습니다."

    valid_df.to_csv(f"{save_path}/valid_data.csv", index=0, encoding="utf-8-sig")
    remained_train_data.to_csv(
        f"{save_path}/train_data.csv", index=0, encoding="utf-8-sig"
    )
    print(f"{save_path}에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="split data to train & valid and save")
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./")

    args = parser.parse_args()

    split_dataset(result_path=args.result_path, save_path=args.save_path)
