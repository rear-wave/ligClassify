import numpy as np
import pytest

from data.manifest import PieceTable, SourceRecord


@pytest.fixture
def single_stratum_table(tmp_path):
    source = SourceRecord(
        path=str(tmp_path / "NCG" / "day" / "0-100km" / "one.lig"),
        relative_path="NCG/day/0-100km/one.lig",
        type_index=1,
        distance_bin=0,
        piece_count=20,
    )
    return PieceTable(
        root=str(tmp_path),
        sources=(source,),
        source_index=np.zeros(20, dtype=np.int32),
        piece_index=np.arange(20, dtype=np.int32),
        type_index=np.ones(20, dtype=np.int8),
        distance_bin=np.zeros(20, dtype=np.int8),
        daylight=np.ones(20, dtype=bool),
        timestamp_seconds=np.arange(20, dtype=np.int64),
    )


@pytest.fixture
def piece_table(tmp_path):
    per_type = 40
    sources = tuple(
        SourceRecord(
            path=str(tmp_path / f"type_{type_index}.lig"),
            relative_path=f"type_{type_index}.lig",
            type_index=type_index,
            distance_bin=-1 if type_index == 0 else type_index,
            piece_count=per_type,
        )
        for type_index in range(5)
    )
    type_index = np.repeat(np.arange(5, dtype=np.int8), per_type)
    return PieceTable(
        root=str(tmp_path),
        sources=sources,
        source_index=np.repeat(np.arange(5, dtype=np.int32), per_type),
        piece_index=np.tile(np.arange(per_type, dtype=np.int32), 5),
        type_index=type_index,
        distance_bin=np.where(type_index == 0, -1, type_index).astype(np.int8),
        daylight=np.tile(np.arange(per_type) % 2 == 0, 5),
        timestamp_seconds=np.arange(5 * per_type, dtype=np.int64),
    )
