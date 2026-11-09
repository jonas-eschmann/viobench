"""Published episode sets: conta references of episode archives (one .zip per episode, docs/DATA.md).

    viobench evaluate solution.wasm          # evaluates `public`, downloaded once into the conta cache
    viobench evaluate solution.wasm conta:<sha1> data/val

A set is a list of {"description", "hash"} entries. The hash, the sha1 of an archive, is the episode: reports, logs
and the server know it by nothing else. Descriptions only label conta's log lines and may conceal source filenames.
`SETS` names presets of episode hashes; every selected episode contributes to one aggregate score.
The submission server uses these sets as named presets; its current ordered selection is stored in SQLite and
managed by admins through the UI/API.
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

# Additional evaluation episodes, in catalog order; source filenames are intentionally omitted.
private = [
    {"description": "private/episode-0000.zip", "hash": "53b9f08015131a05be70a386d3d17ac83d9a3807"},
    {"description": "private/episode-0001.zip", "hash": "bc763bd0d9163268b4acc79dc9e3408910ff087a"},
    {"description": "private/episode-0002.zip", "hash": "43623345d941aaa95e5305d77434cfbf71f8d2a3"},
    {"description": "private/episode-0003.zip", "hash": "16ec056c8c440e199d620780a04a591c2b444f05"},
    {"description": "private/episode-0004.zip", "hash": "5404da24acbf42551ee929b74ee69773e8d8f875"},
    {"description": "private/episode-0005.zip", "hash": "47585ad5fed42c8cf5d6e392cb12a3d53fc204b4"},
    {"description": "private/episode-0006.zip", "hash": "966666ab12d925be91737194e272cd0ed3575c1d"},
    {"description": "private/episode-0007.zip", "hash": "5e71688fa1fe2dcd5ff1470aadb2d13d6565bca9"},
    {"description": "private/episode-0008.zip", "hash": "84f5043f2572c22ff3d5832e51ce75448ace7d4e"},
    {"description": "private/episode-0009.zip", "hash": "1b38f5e9c6a880f90208439f764326c10110ede4"},
    {"description": "private/episode-0010.zip", "hash": "b279397f8716e7747b65aa69e40aa746f4b98130"},
    {"description": "private/episode-0011.zip", "hash": "aae136ee7d2a71289fc17a7085281bc385e93590"},
    {"description": "private/episode-0012.zip", "hash": "824adabc69acecdafe2d71e840992e71cf9e9831"},
    {"description": "private/episode-0013.zip", "hash": "b1012aa11bf8a72834e07d9a45d0ed2bd3977ac0"},
]

SETS = {"public": public, "private": private, "public+private-10": public + private[:10]}
