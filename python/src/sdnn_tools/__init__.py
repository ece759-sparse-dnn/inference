"""Utilities for Sparse DNN Graph Challenge."""

import argparse
from pathlib import Path

from . import constants, convert, download, layout


def main() -> None:
    parser = argparse.ArgumentParser(prog="sdnn-tools")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (default: $SDNN_DATA_DIR, else <repo>/data)",
    )

    subparsers = parser.add_subparsers(
        title="subcommands", description="Available subcommands", required=True
    )

    download_parser = subparsers.add_parser(
        "download", help="Download graph challenge datasets into <data>/raw"
    )
    download_parser.add_argument(
        "-s",
        type=int,
        nargs="+",
        help="Size(s) of dataset (Neurons) to download data for",
        default=[1024],
        choices=constants.SIZES,
        metavar="SIZE",
    )
    download_parser.set_defaults(func=download.download)

    convert_parser = subparsers.add_parser(
        "convert", help="Serialize <data>/raw TSVs into <data>/bin .npy files"
    )
    convert_parser.add_argument(
        "-s",
        type=int,
        nargs="+",
        help="Size(s) of dataset (Neurons) to convert data for",
        default=[1024],
        choices=constants.SIZES,
        metavar="SIZE",
    )
    convert_parser.add_argument(
        "--force", action="store_true", help="Re-convert even if already present"
    )
    convert_parser.set_defaults(func=convert.convert)

    args: argparse.Namespace = parser.parse_args()
    layout.set_data_root(args.data_dir)
    args.func(args)
