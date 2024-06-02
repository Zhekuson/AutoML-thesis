import argparse
import os

import wandb
from config_parsers.parsers import ConfigParser
from pipelines.common import ResearchPipeline

wandb.login(key=os.environ["wandb_key"])
print("Config path: ", os.environ["config_path"])

if __name__ == "__main__":
    print("RUNNING")
    parser = argparse.ArgumentParser(description='Running')
    parser.add_argument("--config_path", "-c", "-p", metavar="c", type=str, dest="config_path")
    config_path = parser.parse_args().config_path

    cp = ConfigParser(config_path)
    pipeline = ResearchPipeline(
        **cp.parse()
    )
    pipeline.run()


