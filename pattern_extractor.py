import re


class Extractor:
    def get_match_pattern_string(
        self,
        total_page: str,
        head_string: str,
        target_string: str,
        tail_string: str,
    ):
        pattern = head_string + target_string + tail_string
        total_pattern = re.findall(pattern, total_page)

        return total_pattern
