"""Generate sample STEP parts for testing moldflow.

Run:  python samples/make_samples.py
"""
from pathlib import Path
import cadquery as cq

OUT = Path(__file__).parent


def plate_with_hole():
    """100 x 60 x 2.5 mm plate with a 16 mm hole - classic weld line behind the hole."""
    p = cq.Workplane("XY").box(100, 60, 2.5).faces(">Z").workplane().hole(16)
    return p


def ribbed_housing():
    """Shallow open box with ribs, a boss and a thick corner - race-tracking & warpage."""
    w = (cq.Workplane("XY").box(120, 80, 20)
         .faces(">Z").shell(-2.0))                                   # 2 mm walls, open top
    # ribs on the floor
    ribs = (cq.Workplane("XY").workplane(offset=-10 + 2)
            .rect(100, 60, forConstruction=True).vertices()
            .rect(1.2, 30).extrude(8))
    rib2 = cq.Workplane("XY").workplane(offset=-10 + 2).center(0, 0).rect(100, 1.2).extrude(6)
    # cylindrical boss with a 4 mm core hole
    boss = (cq.Workplane("XY").workplane(offset=-10 + 2).center(30, 0)
            .circle(6).circle(2).extrude(12))
    # local thick pad (warpage driver)
    pad = cq.Workplane("XY").workplane(offset=-10 + 2).center(-35, -20).rect(20, 20).extrude(2.5)
    hole = cq.Workplane("XY").workplane(offset=-10 + 2).center(-30, 15).circle(5).extrude(-5)
    part = w.union(ribs).union(rib2).union(boss).union(pad).cut(hole)
    return part


def bracket():
    """L bracket with two mounting holes and a fillet - two-gate weld test."""
    base = cq.Workplane("XY").box(80, 40, 3)
    wall = cq.Workplane("XY").box(80, 3, 40).translate((0, -18.5, 18.5))
    part = base.union(wall)
    part = part.faces(">Z").workplane().pushPoints([(-25, 5), (25, 5)]).hole(6)
    part = part.faces(">Y").workplane().pushPoints([(0, 18.5 + 10)]).hole(8)
    return part


if __name__ == "__main__":
    for name, fn in [("plate_with_hole", plate_with_hole), ("ribbed_housing", ribbed_housing), ("bracket", bracket)]:
        path = OUT / f"{name}.step"
        cq.exporters.export(fn(), str(path))
        print("wrote", path)
