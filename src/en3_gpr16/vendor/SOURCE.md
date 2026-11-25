# Frozen numerical source

These two files are **unmodified copies** from the completed run
`EN3_GPR16_20261124_102844/source_snapshot/`:

| File | SHA-256 |
| --- | --- |
| `train_gpr_from_pose_data.py` | `8be6c3cab48bc05194b3d153ab43216aad9382e1547bda6c31d3f981038529a8` |
| `validator_module1_learning.py` | `74dc855a5e00bf1e952c2cf71efaa50a51c24d61b0b9649d9b3eee5f0d74de6d` |

The portable adapter calls the original trainer's sample constructor, GP
builder/training function, prediction function, metrics and CSV writers.
The validator checks the complete schema-v5 event records. Its legacy
four-boundary training selection is discarded: the adapter reconstructs the
same **three-boundary, single-event** windows as the frozen EN3 run.

Use the repository's `scripts/train.py` and `scripts/evaluate.py` entry points.
The vendor trainer's original standalone CLI includes other split, sampling
and calibration modes; those are not the EN3 separate-session protocol.
The validator also contains older physical-memory algorithms; the EN3
adapter uses only `EventLearning(8)` for record validation, and does not
train or invoke those models.

No Slicer GUI, Qt, VTK, serial, NDI tracker, or robot-control module is imported
by the portable entry points or these vendored numerical modules.
