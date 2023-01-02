import argparse
import re

import requests

from pattern_extractor import Extractor


def song_crawl(url):
    # youtube url
    page_url = url

    page = requests.get(page_url)
    page_text = page.text
    extractor = Extractor(total_page=page_text)

    text_pattern = '(?<="text"\:").*?' + '(?=")'
    artist_pattern = extractor.get_match_pattern_string(
        head_string='(?<=\{"infoRowRenderer"\:\{"title"\:\{"simpleText"\:"',
        target_string="아티스트",
        tail_string='"\}).*?' + "\}\}\}\]\}",
    )
    title_pattern = (
        '(?<=\{"compactVideoRenderer"\:\{"title"\:\{"runs"\:\[\{"text"\:").*?'
        + '(?=",)'
    )

    artist_string_list = [
        re.search(text_pattern, _artist_string).group()
        for _artist_string in artist_pattern
    ]
    title_string_list = re.findall(title_pattern, page_text)

    result = list(zip(artist_string_list, title_string_list))
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawler Argparse")
    parser.add_argument(
        "--url", type=str, default="https://www.youtube.com/watch?v=x7NhlDkCrtA&t=1s"
    )
    args = parser.parse_args()

    song_crawl(args.url)
