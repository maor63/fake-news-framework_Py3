'''
@author: Jorge Bendahan jorgeaug@post.bgu.ac.il
'''
from dataset_builder.graph_builder import GraphBuilder

class GraphBuilder_Citation(GraphBuilder):
    """Generate directed graphs where nodes represent authors and edges are citations.
    There exists an edge from author A to author B
    If a post of  A  references a post of B """

    def __init__(self, db):
        GraphBuilder.__init__(self, db)

    def execute(self, window_start):
        author_citations = self._db.get_citations(self._domain)
        if self._num_of_random_authors_for_graph is not None:

            post_id_random_author_guid_dict = self._db.get_post_id_random_author_guid_dictionary()

            random_author_citations = []
            for author_citation in author_citations:
                from_author_guid = author_citation[0]
                to_author_guid = author_citation[1]

                if from_author_guid in post_id_random_author_guid_dict and to_author_guid in post_id_random_author_guid_dict:
                    random_author_citations.append(author_citation)
            author_citations = random_author_citations

        if len(author_citations) > 0:
            self._fill_author_connections(author_citations)


