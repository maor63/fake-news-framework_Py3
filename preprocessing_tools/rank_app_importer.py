from commons.commons import cleaner
from preprocessing_tools.app_importer import AppImporter


class RankAppImporter(AppImporter):
    def __init__(self, db):
        AppImporter.__init__(self, db)
        self._rank_threshold = self._config_parser.eval(self.__class__.__name__, "rank_threshold")

    def _get_clean_content_from_row(self, row):
        rank_end_position = row.find('\t')
        if rank_end_position != -1:
            self.rank = float(row[:rank_end_position])
            content = row[rank_end_position + 1:-1]
        return str(cleaner(content))

    def _add_to_posts_dicts(self, post_dict):
        if self.rank <= self._rank_threshold:
            AppImporter._add_to_posts_dicts(self, post_dict)
