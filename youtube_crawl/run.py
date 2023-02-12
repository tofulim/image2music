import argparse
import os

import pandas as pd
from pattern_extractor import Extractor
from playlist_crawler import playlist_crawl
from song_crawler import song_crawl


def runner(
    extractor: Extractor,
    target_youtuber_tag: str,
    include_string: str,
    save_path: str,
):
    row_data_list = list()
    playlist_outputs = playlist_crawl(target_youtuber_tag, include_string, extractor)

    for playlist_output in playlist_outputs:
        video_id = playlist_output["video_id"]
        playlist_title = playlist_output["playlist_title"]
        thumbnail_list = playlist_output["thumbnail_list"]
        big_thumbnail = thumbnail_list[-1]

        print("*" * 30)
        print("now running ...")
        print(f"video_id: {video_id}")
        print(f"playlist_title: {playlist_title}")
        song_outputs = song_crawl(video_id, extractor)
        if not song_outputs:
            continue

        for song_output in song_outputs:
            artist, title = song_output["artist"], song_output["title"]

            try:
                row_data = (video_id, playlist_title, big_thumbnail, artist, title)
                row_data_list.append(row_data)
            except Exception as e:
                print(f"error occur at row append : {e}")

    playlist_dataframe = pd.DataFrame(
        columns=["video_id", "playlist_title", "thumbnail_url", "artist", "title"],
        data=row_data_list,
    )

    print(f"youtuber {target_youtuber_tag} is done ...")
    print(f"save as {save_path}/{target_youtuber_tag}.csv")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"path {save_path} doesn't exist. now we making ...")

    playlist_dataframe.to_csv(
        f"{save_path}/{target_youtuber_tag}.csv", index=0, encoding="utf-8-sig"
    )
    print("save done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Playlist Crawler Argparse")
    parser.add_argument("--target_youtuber_tag", type=str, default=None)
    parser.add_argument("--include_string", type=str, default="[playlist]")

    args = parser.parse_args()
    extractor = Extractor()

    if not args.target_youtuber_tag:
        youtube_tags = open("assets/youtube_tag.txt", "r").readlines()
        youtube_tags = list(map(lambda x: x.replace("\n", ""), youtube_tags))

        for youtube_tag in youtube_tags:
            runner(
                extractor=extractor,
                target_youtuber_tag=youtube_tag,
                include_string=args.include_string,
                save_path="result",
            )
    else:
        runner(
            extractor=extractor,
            target_youtuber_tag=args.target_youtuber_tag,
            include_string=args.include_string,
            save_path="result",
        )
