"""Published episode sets: conta references of episode archives (one .zip per episode, docs/DATA.md).

    viobench evaluate solution.wasm          # evaluates `public`, downloaded once into the conta cache
    viobench evaluate solution.wasm conta:<sha1> data/val

A set is a list of {"description", "hash"} entries (the sha1 of an archive; conta prints the description in its log
lines): the `metadata.json` of the conta shard that holds the archives, copied verbatim. `SETS` names the sets, and that
name is the dataset of their episodes in reports, however a reference to them is spelled. The submission server
evaluates `viobench.tools.server.worker.EPISODES`.
"""

# conta shard viobench-public: the validation episodes, HyperDrone flights through ProcTHOR-Train-1
public = [
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0000.zip", "hash": "0386e5c941b50fe9972ddabff84c414540fb8989"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0001.zip", "hash": "9b6a92791e46815931b6d6f20e74ca9ebf62682e"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0002.zip", "hash": "6e8e3766f46272c1b91eca39c320081251884947"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0003.zip", "hash": "f8fb89fff0c39d51e0f073603ef93b9a2ee00455"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0004.zip", "hash": "489ad518363d88bfa877f86afea144de0d846258"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0005.zip", "hash": "51ea6311a50f4eeb59672012b9796244dab52f60"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0006.zip", "hash": "60ed4323f3aab9c83a854f30062cb4ffbefeac12"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0007.zip", "hash": "8fff5c8dc6376171fc6f865fd214cee44df2b7c3"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0008.zip", "hash": "41e11443a9a5cc165ef9aeeaef11dba63614685b"},
    {"description": "viobench-public/sim-ProcTHOR-Train-1-0009.zip", "hash": "8321da1faa306e72df46f86a72aa3fa0d611c6d9"},
]

SETS = {"public": public}
