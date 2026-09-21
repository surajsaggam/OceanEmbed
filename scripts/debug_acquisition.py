"""
scripts/debug_acquisition.py
----------------------------
Diagnostics for Copernicus Marine DUACS SSH/SLA and GLORYS depth levels.
"""

from __future__ import annotations

import copernicusmarine
from utils.credentials import get_copernicus_credentials, load_env


def test_ssh_dataset():
    load_env()
    user, pwd = get_copernicus_credentials()
    dataset_id = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D"

    print(f"\n--- Testing SSH/SLA Dataset: {dataset_id} ---")
    try:
        # Try dry_run subset
        sub = copernicusmarine.subset(
            dataset_id=dataset_id,
            username=user,
            password=pwd,
            variables=["sla", "adt", "ugosa", "vgosa"],
            minimum_longitude=45.0,
            maximum_longitude=105.0,
            minimum_latitude=5.0,
            maximum_latitude=30.0,
            start_datetime="2020-01-01",
            end_datetime="2020-01-02",
            dry_run=True,
        )
        print("Dry run subset succeeded with all 4 variables!")
    except Exception as e:
        print(f"Failed with 4 variables: {e}")
        # Try with just 'sla'
        try:
            sub = copernicusmarine.subset(
                dataset_id=dataset_id,
                username=user,
                password=pwd,
                variables=["sla"],
                minimum_longitude=45.0,
                maximum_longitude=105.0,
                minimum_latitude=5.0,
                maximum_latitude=30.0,
                start_datetime="2020-01-01",
                end_datetime="2020-01-02",
                dry_run=True,
            )
            print("Dry run subset succeeded with ['sla']!")
        except Exception as e2:
            print(f"Failed with ['sla']: {e2}")


def test_glorys_depths():
    load_env()
    user, pwd = get_copernicus_credentials()
    dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

    print(f"\n--- Testing GLORYS Depth Levels: {dataset_id} ---")
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            username=user,
            password=pwd,
        )
        depths = ds.depth.values
        print(f"Total GLORYS depth levels: {len(depths)}")
        print("Depths around 1000m:")
        for idx, d in enumerate(depths):
            if 700 <= d <= 1200:
                print(f"  index {idx}: {d:.3f} m")
    except Exception as e:
        print(f"Failed to open GLORYS dataset: {e}")


if __name__ == "__main__":
    test_ssh_dataset()
    test_glorys_depths()
