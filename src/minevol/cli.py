"""Command line: ``minevol <stage> [survey] --config config/john_henry.yaml``."""

from __future__ import annotations

import argparse
import logging

from . import workflow as wf


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="minevol", description=__doc__)
    ap.add_argument("stage", choices=["fetch", "process", "toes", "volumes", "resolution",
                                      "change", "report", "all"])
    ap.add_argument("survey", nargs="?", help="survey key from the config (default: all)")
    ap.add_argument("--config", default="config/john_henry.yaml")
    ap.add_argument("--ground", default="vendor", choices=["vendor", "smrf"],
                    help="ground classification to use (default: the vendor's)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    prj = wf.Project.load(a.config)
    surveys = [a.survey] if a.survey else list(prj.cfg["surveys"])

    # The change stage runs before volumes: its stable-ground variogram sets the
    # survey error that volumes() propagates (see workflow._survey_error).
    stages = ["fetch", "process", "change", "volumes", "resolution", "report"] \
        if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "change":
            print(wf.change(prj, a.ground))
            continue
        if st == "report":
            from .report import build_report
            build_report(prj)
            continue
        for s in surveys:
            logging.info("%s: %s", st, s)
            if st == "fetch":
                wf.fetch(prj, s)
            elif st == "process":
                print(wf.process(prj, s, a.ground))
            elif st == "toes":
                print(wf.toes(prj, s, a.ground).drop(columns="geometry").to_string())
            elif st == "volumes":
                print(wf.volumes(prj, s, a.ground).to_string())
            elif st == "resolution":
                print(wf.resolution_study(prj, s).to_string())


if __name__ == "__main__":
    main()
