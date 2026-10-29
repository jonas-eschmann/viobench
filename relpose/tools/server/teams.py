"""Team tokens for the submission server, kept in the server's sqlite. A team is its token; the name is a note for you.

    python -m relpose.tools.server.teams add "team a" "team b"   # prints "<token>  <name>" for each new team
    python -m relpose.tools.server.teams list                     # every team's token and name

A token is 24 random bytes as 32 base64url characters. Hand each team its own: it goes in every request as
"Authorization: Bearer <token>". Uses RELPOSE_SERVER_DIR like the app and the worker.
"""
import argparse

from .store import Store


def main():
    parser = argparse.ArgumentParser(prog="python -m relpose.tools.server.teams", description="Create and list team tokens.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="create teams and print their tokens").add_argument("names", nargs="+")
    commands.add_parser("list", help="print every team's token and name")
    arguments = parser.parse_args()
    store = Store()
    if arguments.command == "add":
        for name in arguments.names:
            print(f"{store.add_team(name)}  {name}")
    else:
        for token, name in store.teams():
            print(f"{token}  {name}")


if __name__ == "__main__":
    main()
