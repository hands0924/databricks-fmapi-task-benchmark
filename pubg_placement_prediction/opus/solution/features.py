"""Feature engineering for PUBG winPlacePerc prediction.

Target is constant per (matchId, groupId) and lies exactly on the
1/(maxPlace-1) grid, so we model at GROUP level and post-process.
"""
import numpy as np
import pandas as pd

# columns that are per-match constants (must not be aggregated as behaviour)
MATCH_CONST = ["matchDuration", "maxPlace", "numGroups"]
DROP = ["Id", "groupId", "matchId", "matchType", "winPlacePerc"]


def add_player_features(df):
    """Row-level (per-player) derived features."""
    df = df.copy()
    eps = 1e-6

    df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["healsBoosts"] = df["heals"] + df["boosts"]
    df["items"] = df["heals"] + df["boosts"] + df["weaponsAcquired"]
    df["killsAssists"] = df["kills"] + df["assists"]
    df["killPlacePerc"] = df["killPlace"] / df["maxPlace"].clip(lower=1)
    df["headshotRate"] = df["headshotKills"] / (df["kills"] + eps)
    df["killStreakRate"] = df["killStreaks"] / (df["kills"] + eps)
    df["damagePerKill"] = df["damageDealt"] / (df["kills"] + eps)
    df["walkPerDuration"] = df["walkDistance"] / (df["matchDuration"] + eps)
    df["distancePerDuration"] = df["totalDistance"] / (df["matchDuration"] + eps)
    df["killsPerDistance"] = df["kills"] / (df["totalDistance"] + 1.0)
    df["damagePerDistance"] = df["damageDealt"] / (df["totalDistance"] + 1.0)
    df["healsPerDistance"] = df["healsBoosts"] / (df["totalDistance"] + 1.0)
    df["itemsPerDistance"] = df["items"] / (df["totalDistance"] + 1.0)
    df["weaponsPerDistance"] = df["weaponsAcquired"] / (df["totalDistance"] + 1.0)
    df["killsNoMove"] = ((df["kills"] > 0) & (df["totalDistance"] == 0)).astype(np.int8)
    df["boostsPerWalk"] = df["boosts"] / (df["walkDistance"] + 1.0)
    df["longestKillPerKill"] = df["longestKill"] / (df["kills"] + eps)
    df["skill"] = df["headshotKills"] + df["roadKills"] * 5
    df["teamworkScore"] = df["revives"] + df["assists"] - df["teamKills"] * 2
    df["maxPlaceOverNumGroups"] = df["maxPlace"] / df["numGroups"].clip(lower=1)

    # rankPoints is -1 when unused; killPoints/winPoints are 0 when unused
    df["rankPointsFix"] = df["rankPoints"].replace(-1, np.nan)
    df["killPointsFix"] = df["killPoints"].replace(0, np.nan)
    df["winPointsFix"] = df["winPoints"].replace(0, np.nan)
    return df


def _match_type_group(s):
    s = s.astype(str)
    out = pd.Series(3, index=s.index, dtype=np.int8)  # other
    out[s.str.contains("solo")] = 0
    out[s.str.contains("duo")] = 1
    out[s.str.contains("squad")] = 2
    return out


def build_group_features(df, is_train=True):
    """Aggregate to (matchId, groupId) rows with match-relative rank features."""
    df = add_player_features(df)

    df["mt_kind"] = _match_type_group(df["matchType"])
    df["mt_fpp"] = df["matchType"].astype(str).str.contains("fpp").astype(np.int8)
    df["mt_normal"] = df["matchType"].astype(str).str.contains("normal").astype(np.int8)

    feats = [c for c in df.columns if c not in DROP and c not in
             ("mt_kind", "mt_fpp", "mt_normal")]
    behaviour = [c for c in feats if c not in MATCH_CONST]

    keys = ["matchId", "groupId"]
    g = df.groupby(keys, sort=False)

    out = g.size().to_frame("groupSize")
    # match-level metadata (constant inside a match)
    meta = g[MATCH_CONST + ["mt_kind", "mt_fpp", "mt_normal"]].first()
    out = out.join(meta)

    pieces = [out]
    for how in ["mean", "max", "min", "sum"]:
        a = g[behaviour].agg(how)
        a.columns = [f"{c}_{how}" for c in a.columns]
        pieces.append(a.astype(np.float32))
    # std only meaningful for multi-player groups
    a = g[behaviour].std()
    a.columns = [f"{c}_std" for c in a.columns]
    pieces.append(a.astype(np.float32))

    grp = pd.concat(pieces, axis=1)
    del pieces

    if is_train:
        grp["winPlacePerc"] = g["winPlacePerc"].first()

    grp = grp.reset_index()

    # ---- match-relative features -------------------------------------------
    m = grp.groupby("matchId", sort=False)
    grp["numGroupsActual"] = m["groupId"].transform("size").astype(np.float32)
    grp["playersInMatch"] = m["groupSize"].transform("sum").astype(np.float32)
    grp["groupSizeRatio"] = grp["groupSize"] / grp["playersInMatch"]
    grp["numGroupsRatio"] = grp["numGroupsActual"] / grp["maxPlace"].clip(lower=1)

    rank_src = [c for c in grp.columns
                if c.endswith(("_mean", "_max", "_min", "_sum"))]
    rank_blocks = []
    r = grp.groupby("matchId", sort=False)[rank_src].rank(pct=True, method="average")
    r.columns = [f"{c}_rank" for c in rank_src]
    rank_blocks.append(r.astype(np.float32))

    # deviation from match mean for the group-mean features
    mean_src = [c for c in grp.columns if c.endswith("_mean")]
    mm = grp.groupby("matchId", sort=False)[mean_src].transform("mean")
    d = (grp[mean_src].values - mm.values) / (np.abs(mm.values) + 1e-6)
    d = pd.DataFrame(d, index=grp.index,
                     columns=[f"{c}_mdev" for c in mean_src]).astype(np.float32)
    rank_blocks.append(d)

    grp = pd.concat([grp] + rank_blocks, axis=1)
    return grp


def feature_columns(grp):
    excl = {"matchId", "groupId", "winPlacePerc"}
    return [c for c in grp.columns if c not in excl]
