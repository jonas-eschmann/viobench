"""Team and admin tokens for the submission server, kept in the server's sqlite.

    python -m viobench.tools.server.teams add --type participant "team a" "team b"  # prints "<token>  <name>" for each participant
    python -m viobench.tools.server.teams add "test"              # a test account, not a participant
    python -m viobench.tools.server.teams add --unlimited "debug"  # a team without a daily upload quota
    python -m viobench.tools.server.teams add --type admin "admin"      # view all submissions and pause new uploads
    python -m viobench.tools.server.teams add --type baseline "reference"  # share scores in comparisons
    python -m viobench.tools.server.teams update "$TOKEN" --type participant  # change an existing account's type
    python -m viobench.tools.server.teams list                     # every team's token and name

A token is 24 random bytes as 32 base64url characters. Hand each team its own: it goes in every request as
"Authorization: Bearer <token>". Uses VIOBENCH_SERVER_DIR like the app and the worker.
"""
import argparse

from .store import TEAM_TYPES, Store


def main():
    parser = argparse.ArgumentParser(prog="python -m viobench.tools.server.teams", description="Create, update and list team and admin accounts.")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create teams and print their tokens")
    add.add_argument("names", nargs="+")
    add.add_argument("--unlimited", action="store_true", help="no daily quota, e.g. for the instructors' debug team")
    add.add_argument("--type", choices=TEAM_TYPES, default="test", help="account type (default: test)")
    update = commands.add_parser("update", help="change an existing team's account type")
    update.add_argument("token", help="the team's token, as shown by list")
    update.add_argument("--type", choices=TEAM_TYPES, required=True, help="new account type")
    commands.add_parser("list", help="print every team's token and name")
    arguments = parser.parse_args()
    store = Store()
    if arguments.command == "add":
        for name in arguments.names:
            token = store.add_team(name, team_type=arguments.type, unlimited=arguments.unlimited)
            print(f"{token}  {name}")
    elif arguments.command == "update":
        try:
            store.update_team(arguments.token, team_type=arguments.type)
        except ValueError as error:
            parser.error(str(error))
        print("Updated team.")
    else:
        for token, name, unlimited, kind in store.teams():
            print(f"{token}  {name}  ({kind}){'  (unlimited)' if unlimited else ''}")


if __name__ == "__main__":
    main()
