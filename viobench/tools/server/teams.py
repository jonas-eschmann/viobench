"""Team tokens for the submission server, kept in the server's sqlite. A team is its token; the name is a note for you.

    python -m viobench.tools.server.teams add "team a" "team b"   # prints "<token>  <name>" for each new team
    python -m viobench.tools.server.teams add --unlimited "debug"  # a team without a daily quota (still one at a time)
    python -m viobench.tools.server.teams list                     # every team's token and name

A token is 24 random bytes as 32 base64url characters. Hand each team its own: it goes in every request as
"Authorization: Bearer <token>". Uses VIOBENCH_SERVER_DIR like the app and the worker.
"""
import argparse

from .store import Store


def main():
    parser = argparse.ArgumentParser(prog="python -m viobench.tools.server.teams", description="Create and list team tokens.")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create teams and print their tokens")
    add.add_argument("names", nargs="+")
    add.add_argument("--unlimited", action="store_true", help="no daily quota, e.g. for the instructors' debug team")
    commands.add_parser("list", help="print every team's token and name")
    arguments = parser.parse_args()
    store = Store()
    if arguments.command == "add":
        for name in arguments.names:
            print(f"{store.add_team(name, arguments.unlimited)}  {name}")
    else:
        for token, name, unlimited in store.teams():
            print(f"{token}  {name}{'  (unlimited)' if unlimited else ''}")


if __name__ == "__main__":
    main()
