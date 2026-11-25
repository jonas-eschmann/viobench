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

private_stage2 = [
    {"description": "private_stage2/episode-0000.zip", "hash": "004c5894ac968aae213ed3ec57426accb931efb9"},
    {"description": "private_stage2/episode-0001.zip", "hash": "1c6e73f3323e2ed13ace3e9f396fe9198025320f"},
    {"description": "private_stage2/episode-0002.zip", "hash": "21494468052b8747b0c0521d58055c0a40cf9892"},
    {"description": "private_stage2/episode-0003.zip", "hash": "726458d9bc3167a808af77c5a86429f5edbacc1d"},
    {"description": "private_stage2/episode-0004.zip", "hash": "78ed493f5b51b5ef6b985cc6945175b1aa7c85f0"},
    {"description": "private_stage2/episode-0005.zip", "hash": "8950399d97f58ba1195258752dedd3be71980d69"},
    {"description": "private_stage2/episode-0006.zip", "hash": "89f8c99e0a08cb30cab6ad67977dda2d49010244"},
    {"description": "private_stage2/episode-0007.zip", "hash": "a61ad08d7b50a01c3933a3076bcedf5702fd2845"},
    {"description": "private_stage2/episode-0008.zip", "hash": "cd6dc704267af373a3c3404571ac57de1d628bf3"},
    {"description": "private_stage2/episode-0009.zip", "hash": "dd7a737e6813dced18112fb2013c725055b357cb"},
    {"description": "private_stage2/episode-0010.zip", "hash": "e7aa5aee7fb75a9257fd4e1d48d04905db407c74"},
    {"description": "private_stage2/episode-0011.zip", "hash": "f7c2546ff0f292874d371a006eac265cc9e6ae2b"},
]

private_stage3 = [
    {"description": "private_stage3/episode-0000.zip", "hash": "30755ac25f178027de87c789fc84479dddf293f1"},
    {"description": "private_stage3/episode-0001.zip", "hash": "500bc1fdb426fabd63317942060484b2e0fba0d7"},
    {"description": "private_stage3/episode-0002.zip", "hash": "92c12d5a6a048ed0e8b7dd47ef42428acf666c0f"},
    {"description": "private_stage3/episode-0003.zip", "hash": "9d8c612ee5ad775f77f54efbbf542f5b61ac3578"},
    {"description": "private_stage3/episode-0004.zip", "hash": "d6337f1033b1abc6324d3e1f1926937826bdb31c"},
    {"description": "private_stage3/episode-0005.zip", "hash": "9affc13d7522343cfe4edd888c427bcc47fc3f2e"},
    {"description": "private_stage3/episode-0006.zip", "hash": "bffcea9dd7953a3c5f921535603a22192dfcaf70"},
    {"description": "private_stage3/episode-0007.zip", "hash": "3cc7fd3d0baee5564b98681ea2e250eef44d475d"},
    {"description": "private_stage3/episode-0008.zip", "hash": "f794e77752be1200e8cb3c2d815d4d44ef1584ab"},
    {"description": "private_stage3/episode-0009.zip", "hash": "7519dd967783dbfb7d424b535d4b66243bf31596"},
    {"description": "private_stage3/episode-0010.zip", "hash": "e5cc48d7b6d89ce490af00431200a2781fd9c5fb"},
    {"description": "private_stage3/episode-0011.zip", "hash": "293ef2a9a0d0f14f0bec5ec1f3823a5a17b91627"},
    {"description": "private_stage3/episode-0012.zip",  "hash": "5672eb24b81b929174743ef11fee342e38026c04"},
    {"description": "private_stage3/episode-0013.zip",  "hash": "f141220807fd9276019c226bad9cfe3978e04173"},
    {"description": "private_stage3/episode-0014.zip",  "hash": "72bef50e78e672e42c2291784a8857f7fb3e32f6"},
    {"description": "private_stage3/episode-0015.zip",  "hash": "c31ed49113cd798c29ed8e84eaeb293c90955ac6"},
    {"description": "private_stage3/episode-0016.zip",  "hash": "34c68d36d10c489b33a201037242df5469cf0321"},
    {"description": "private_stage3/episode-0017.zip",  "hash": "6fd93fa326fdb4c10159a25f7e5c2647b7857249"},
    {"description": "private_stage3/episode-0018.zip",  "hash": "dc4e9bfa15c8299535b47df63d64f8e52d0ca9ef"},
    {"description": "private_stage3/episode-0019.zip",  "hash": "616b85ed75c0a92c5b4829a3ac2905369eb9706e"},
    {"description": "private_stage3/episode-0020.zip",  "hash": "4fb55ea93a6f22ca59d0bb3db2ffb00beefbd66f"},
    {"description": "private_stage3/episode-0021.zip",  "hash": "9f250c4f6691d11aed2f6a011a64b97b2d7cdb1d"},
    {"description": "private_stage3/episode-0022.zip",  "hash": "1411ebe11cf348657ea4371fd58d2681d01a4d5b"},
    {"description": "private_stage3/episode-0023.zip",  "hash": "ea5ede02321a9d1f92df38efc8abf9b85420ed40"}
]

SETS = {"public": public, "private": private, "public+private-10": public + private[:10], "private_stage2": private_stage2, "private_stage3": private_stage3}
