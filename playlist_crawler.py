import argparse

import requests
from bs4 import BeautifulSoup as bs

from pattern_extractor import Extractor


def get_image(
    page_source: str,
    target_tag: str,
):
    """get image
    get thumbnail of playlist. it's quality may be low
    Args:
        page_source (str): html text of target page
        target_tag (str): target html tag of image thumbnail

    Returns:
        image url list that can access public
    """
    page_source.select_one(target_tag)


def playlist_crawl(
    target_youtuber_tag: str,
    target_string: str,
    extractor: Extractor,
):
    """playlist crawler
    get playlist's youtube video id & image thumbnails
    Args:
        target_youtuber_tag (str): playlist's owner(youtuber tag)
        target_string (str): string that u want to include
        extractor (Extractor):

    Returns:

    """
    # youtube url
    page_url = f"https://www.youtube.com/@{target_youtuber_tag}/videos"
    page = requests.get(page_url, verify=False)
    page_source = bs(page.text, "html.parser")

    video_id_list = extractor.get_match_pattern_string(
        head_string='"content"\:\{"videoRenderer"\:\{"videoId"\:"',
        target_string="",
        tail_string='"\,"thumbnail"\:\{"thumbnails',
        total_page=page_source,
    )
    print(video_id_list)
    print(f"ori len : {len(video_id_list)}")
    new_l = dict.fromkeys(video_id_list)
    print(new_l)
    print(f"ori len : {len(new_l)}")

    # return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawler Argparse")
    parser.add_argument("--target_youtuber_tag", type=str, default="micasfu")
    parser.add_argument("--target_string", type=str, default="[playlist]")

    args = parser.parse_args()

    extractor = Extractor()
    playlist_crawl(args.target_youtuber_tag, args.target_string, extractor)
