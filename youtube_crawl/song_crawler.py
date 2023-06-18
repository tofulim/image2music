import argparse
import re

import requests
from pattern_extractor import Extractor


def song_crawl(video_id: str, extractor: Extractor):
    # youtube url
    page_url = f"https://www.youtube.com/watch?v={video_id}"

    page = requests.get(page_url)
    page_text = page.text

    artist_source = extractor.get_match_pattern_string(
        head_string='infoRowRenderer"\:\{"title"\:\{"simpleText"\:"아티스트"',
        target_string="",
        tail_string="\}\}\}\]\}",
        total_page=page_text,
    )
    title_pattern = (
        '(?<=\{"compactVideoRenderer"\:\{"title"\:\{"runs"\:\[\{"text"\:").*?'
        + '(?=",)'
    )

    artist_string_list = list()
    for _artist_source in artist_source:
        artist = extractor.get_match_pattern_string(
            head_string='simpleText"\:"',
            target_string="",
            tail_string='"\}\,"trackingParams',
            total_page=_artist_source,
        )
        if artist:
            artist = artist[0]
        else:
            artist = extractor.get_match_pattern_string(
                head_string='text"\:"',
                target_string="",
                tail_string='"\,"navigationEndpoint',
                total_page=_artist_source,
            )[0]
        artist_string_list.append(artist)

    title_string_list = re.findall(title_pattern, page_text)

    print(f"artist list : {artist_string_list}")
    print(f"title_string_list list : {title_string_list}")

    if len(artist_string_list) != len(title_string_list):
        print(
            f"artist len{len(artist_string_list)} doesn't match with title len{len(title_string_list)}"
        )

        return None

    result = [
        {
            "artist": artist,
            "title": title,
        }
        for artist, title in list(zip(artist_string_list, title_string_list))
    ]

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawler Argparse")
    parser.add_argument("--video_id", type=str, default="4GsGfgMRe6s")
    args = parser.parse_args()

    extractor = Extractor()

    result = song_crawl(args.video_id, extractor)
    print(result)
