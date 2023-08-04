import argparse

import requests
from pattern_extractor import Extractor


def playlist_crawl(
    target_youtuber_tag: str,
    include_string: str,
    extractor: Extractor,
):
    """playlist crawler
    get playlist's youtube video id & image thumbnails
    Args:
        target_youtuber_tag (str): playlist's owner(youtuber tag)
        target_string (str): string that u want to include
        extractor (Extractor):

    Returns:
        video_information (list): consist of dict that has keys(video_id(str), title(str), thunbnail_list(list))

    """
    video_information_list = list()

    # youtube url
    page_url = f"https://www.youtube.com/@{target_youtuber_tag}/videos"
    page = requests.get(page_url, verify=False)
    page_source = page.text

    # get target sub_string from total_string which has all data
    target_video_sources = extractor.get_match_pattern_string(
        head_string="richItemRenderer",
        target_string="",
        tail_string="accessibility",
        total_page=page_source,
    )

    for target_video_source in target_video_sources:
        try:
            video_id = extractor.get_match_pattern_string(
                head_string='videoId"\:"',
                target_string="",
                tail_string='"\,"',
                total_page=target_video_source,
            )
            title = extractor.get_match_pattern_string(
                head_string='text"\:"',
                target_string="",
                tail_string='"\}\]',
                total_page=target_video_source,
            )
            title = list(filter(lambda x: include_string in x.lower(), title))
            thumbnail_list = extractor.get_match_pattern_string(
                head_string='url"\:"',
                target_string="",
                tail_string='"\,"',
                total_page=target_video_source,
            )

            video_information = {
                "video_id": video_id[0],
                "playlist_title": title[0],
                "thumbnail_list": thumbnail_list,
            }

            video_information_list.append(video_information)
        except Exception as e:
            print(f"error occur : {e}")

    assert (
        video_id or title or thumbnail_list
    ), f"len doesn't match video_id {len(video_id)}, title {len(title)}, thumbnail_list {len(thumbnail_list)}"

    return video_information_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawler Argparse")
    parser.add_argument("--target_youtuber_tag", type=str, default="micasfu")
    parser.add_argument("--include_string", type=str, default="[playlist]")

    args = parser.parse_args()

    extractor = Extractor()
    result = playlist_crawl(args.target_youtuber_tag, args.include_string, extractor)
    print(f"total result: {result}")
    print(f"result[0]: {result[0]}")
    print(f"len of result is : {len(result)}")
