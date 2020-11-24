import logging
from configuration.config_class import getConfig
from timeline_overlap_visualization.timeline_overlap_visualization_generator import TimelineOverlapVisualizationGenerator

if __name__ == '__main__':
    config_parser = getConfig()

    timeline_overlap_reporter = TimelineOverlapVisualizationGenerator()
    timeline_overlap_reporter.setUp()
    timeline_overlap_reporter.generate_timeline_overlap_csv()
    print("Finished !!!!")
    logging.info("Finished !!!!")