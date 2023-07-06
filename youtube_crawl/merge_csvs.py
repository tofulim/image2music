import argparse
from glob import glob

import pandas as pd


def merge_csvs(result_path: str, save_path: str):
    """
    playlist.csv result들을 통합하는 함수
    학습을 위한 하나의 csv 파일을 만들어 저장한다

    Args:
        result_path (str): csv 파일들이 들어있는 디렉토리
        save_path (str): 통합한 csv 파일을 저장할 위치

    Returns:
        None

    """

    csv_files = glob(f"{result_path}/*.csv")
    print(f"해당 경로 {result_path}에 {len(csv_files)}개의 파일이 있고 이를 통합합니다.")
    total_csv = pd.DataFrame()
    for csv_file in csv_files:
        csv_file = pd.read_csv(csv_file, encoding="utf-8-sig")
        total_csv = pd.concat([total_csv, csv_file]).reset_index(drop=True)

    # 제목이나 가수에 있는 &를 html에서 추출할 때 제대로 가져오지 못한 경우 이를 후처리 해준다
    total_csv["artist"] = list(
        map(lambda x: x.replace("\\u0026", "&"), total_csv["artist"].values)
    )

    total_csv.to_csv(save_path, index=0)
    print(f"{save_path}에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge csv results and save")
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./total_csv.csv")

    args = parser.parse_args()

    merge_csvs(result_path=args.result_path, save_path=args.save_path)
