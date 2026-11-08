"""Team and admin tokens for the submission server, kept in the server's sqlite.

    python -m viobench.tools.server.teams add --participant "team a" "team b"  # prints "<token>  <name>" for each participant
    python -m viobench.tools.server.teams add "test"              # a test account, not a participant
    python -m viobench.tools.server.teams add --unlimited "debug"  # a team without a daily quota (still one at a time)
    python -m viobench.tools.server.teams add --admin "admin"      # view all submissions and pause new uploads
    python -m viobench.tools.server.teams update "$TOKEN" --participant  # mark an existing test account as a participant
    python -m viobench.tools.server.teams update "$TOKEN" --no-admin     # revoke admin access
    python -m viobench.tools.server.teams list                     # every team's token and name

A token is 24 random bytes as 32 base64url characters. Hand each team its own: it goes in every request as
"Authorization: Bearer <token>". Uses VIOBENCH_SERVER_DIR like the app and the worker.
"""
import argparse

from .store import Store


def main():
    parser = argparse.ArgumentParser(prog="python -m viobench.tools.server.teams", description="Create, update and list team and admin accounts.")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="create teams and print their tokens")
    add.add_argument("names", nargs="+")
    add.add_argument("--unlimited", action="store_true", help="no daily quota, e.g. for the instructors' debug team")
    kind = add.add_mutually_exclusive_group()
    kind.add_argument("--admin", action="store_true", help="view every team's submissions and pause new uploads")
    kind.add_argument("--participant", action="store_true", help="mark a regular participant (otherwise a test account)")
    update = commands.add_parser("update", help="set or clear an existing team's account flags")
    update.add_argument("token", help="the team's token, as shown by list")
    update.add_argument("--admin", action=argparse.BooleanOptionalAction, default=None, help="grant or revoke admin access")
    update.add_argument("--participant", action=argparse.BooleanOptionalAction, default=None, help="set or clear participant status")
    commands.add_parser("list", help="print every team's token and name")
    arguments = parser.parse_args()
    store = Store()
    if arguments.command == "add":
        for name in arguments.names:
            token = store.add_team(name, unlimited=arguments.unlimited, is_admin=arguments.admin, is_participant=arguments.participant)
            print(f"{token}  {name}")
    elif arguments.command == "update":
        try:
            store.update_team(arguments.token, is_admin=arguments.admin, is_participant=arguments.participant)
        except ValueError as error:
            parser.error(str(error))
        print("Updated team.")
    else:
        for token, name, unlimited, is_admin, is_participant in store.teams():
            kind = "admin" if is_admin else "participant" if is_participant else "test"
            print(f"{token}  {name}  ({kind}){'  (unlimited)' if unlimited else ''}")


if __name__ == "__main__":
    main()
