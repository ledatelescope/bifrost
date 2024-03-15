# github_stats.py – Grab a snapshot of github stats
# usage: [-h] --token-file PATH [--repo OWNER/REPO] [--output PATH]

# Reads github user token from token-file path and appends one line of JSON to
# output path.  Default repo is ledatelescope/bifrost.  Default output is
# stdout.  Accumulated log file is JSON-lines format <https://jsonlines.org/>
# and can be queried using `jq`.  Meant to be run daily.

# Prerequisites: python3, PyGithub

from datetime import datetime, timezone
import argparse
import github
import json
import sys


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (github.View.View, github.Clones.Clones)):
            return obj.raw_data
        else:
            return super().default(obj)


def main(token_file, repo, output):
    token = token_file.read().rstrip()
    g = github.Github(token)
    r = g.get_repo(repo)
    snapshot = {
        "timestamp": datetime.now(timezone.utc),
        "repo": r.full_name,
        "stars": r.stargazers_count,
        "watchers": r.watchers_count,
        "forks": r.forks_count,
        "network": r.network_count,
        "open_issues": r.open_issues_count,
        "subscribers": r.subscribers_count,
        "views": r.get_views_traffic(),
        "clones": r.get_clones_traffic(),
    }
    json.dump(snapshot, output, cls=CustomEncoder)
    output.write("\n")


parser = argparse.ArgumentParser(description="Grab snapshot of github stats.")

parser.add_argument(
    "--token-file",
    "-t",
    metavar="PATH",
    required=True,
    type=argparse.FileType("r"),
    help="Read github access token from this file",
)

parser.add_argument(
    "--repo",
    "-r",
    metavar="OWNER/REPO",
    default="ledatelescope/bifrost",
    help="Grab statistics from this repo [ledatelescope/bifrost]",
)

parser.add_argument(
    "--output",
    "-o",
    metavar="PATH",
    default=sys.stdout,
    type=argparse.FileType("a"),
    help="Append data to this file",
)

if __name__ == "__main__":
    main(**vars(parser.parse_args()))
