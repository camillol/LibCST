# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import libcst as cst
from libcst._position import CodePosition, CodeRange
from libcst.metadata.base_provider import BatchableMetadataProvider
from libcst.metadata.file_path_provider import FilePathProvider
from libcst.metadata.position_provider import PositionProvider


class Position(TypedDict):
    line: int
    column: int


class Location(TypedDict):
    path: str
    start: Position
    stop: Position


class InferredType(TypedDict):
    location: Location
    annotation: str


class PyreData(TypedDict, total=False):
    types: Sequence[InferredType]


class TypeInferenceProvider(BatchableMetadataProvider[str]):
    """
    Access inferred type annotation through `Pyre Query API <https://pyre-check.org/docs/querying-pyre.html>`_.
    It requires `setup watchman <https://pyre-check.org/docs/getting-started/>`_
    and start pyre server by running ``pyre`` command.
    The inferred type is a string of `type annotation <https://docs.python.org/3/library/typing.html>`_.
    E.g. ``typing.List[libcst._nodes.expression.Name]``
    is the inferred type of name ``n`` in expression ``n = [cst.Name("")]``.
    All name references use the fully qualified name regardless how the names are imported.
    (e.g. ``import libcst; libcst.Name`` and ``import libcst as cst; cst.Name`` refer to the same name.)
    Pyre infers the type of :class:`~libcst.Name`, :class:`~libcst.Attribute` and :class:`~libcst.Call` nodes.
    The inter process communication to Pyre server is managed by :class:`~libcst.metadata.FullRepoManager`.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    @classmethod
    def gen_cache(
        cls,
        root_path: Path,
        paths: List[str],
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Mapping[str, object]:
        return _gen_rel_path_to_pyre_data_mapping(root_path, paths, timeout)

    def __init__(self, cache: PyreData) -> None:
        super().__init__(cache)
        cache_types = cache.get("types", [])
        self.lookup: Dict[CodeRange, str] = _make_type_lookup(cache_types)

    def _parse_metadata(self, node: cst.CSTNode) -> None:
        range = self.get_metadata(PositionProvider, node)
        if range in self.lookup:
            self.set_metadata(node, self.lookup.pop(range))

    def visit_Name(self, node: cst.Name) -> Optional[bool]:
        self._parse_metadata(node)

    def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
        self._parse_metadata(node)

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        self._parse_metadata(node)

class NonCachedTypeInferenceProvider(BatchableMetadataProvider[str]):
    """
    Access inferred type annotation through `Pyre Query API <https://pyre-check.org/docs/querying-pyre.html>`_.
    It requires `setup watchman <https://pyre-check.org/docs/getting-started/>`_
    and start pyre server by running ``pyre`` command.
    The inferred type is a string of `type annotation <https://docs.python.org/3/library/typing.html>`_.
    E.g. ``typing.List[libcst._nodes.expression.Name]``
    is the inferred type of name ``n`` in expression ``n = [cst.Name("")]``.
    All name references use the fully qualified name regardless how the names are imported.
    (e.g. ``import libcst; libcst.Name`` and ``import libcst as cst; cst.Name`` refer to the same name.)
    Pyre infers the type of :class:`~libcst.Name`, :class:`~libcst.Attribute` and :class:`~libcst.Call` nodes.
    The inter process communication to Pyre server is managed by :class:`~libcst.metadata.FullRepoManager`.
    """

    METADATA_DEPENDENCIES = (PositionProvider, FilePathProvider)


    def visit_Module(self, node: cst.Module) -> None:
        self.path = self.get_metadata(FilePathProvider, node)
        cache = _gen_rel_path_to_pyre_data_mapping_one_path(Path("/"), path=str(self.path), timeout=None)
        cache_types = cache.get(str(self.path), {}).get("types", [])
        self.lookup: Dict[CodeRange, str] = _make_type_lookup(cache_types)

    def _parse_metadata(self, node: cst.CSTNode) -> None:
        range = self.get_metadata(PositionProvider, node)
        if range in self.lookup:
            self.set_metadata(node, self.lookup.pop(range))

    def visit_Name(self, node: cst.Name) -> Optional[bool]:
        self._parse_metadata(node)

    def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
        self._parse_metadata(node)

    def visit_Call(self, node: cst.Call) -> Optional[bool]:
        self._parse_metadata(node)


@lru_cache(maxsize=10)
def _gen_rel_path_to_pyre_data_mapping_one_path(
    root_path: Path, path: str, timeout: Optional[int]
) -> Mapping[str, PyreData]:
    return _gen_rel_path_to_pyre_data_mapping(root_path, [path], timeout)


MAX_ARG_STRLEN=2**17

def _chunked_path_queries(root_path: Path, paths: List[str], limit:int=MAX_ARG_STRLEN):
    BATCH = "batch({})"
    limit -= len(BATCH.format(""))
    arg_chunks: list[str] = []
    chunks_len = 0
    arg_paths: list[str] = []
    for path in paths:
        arg_chunks.append(f"types(path='{(root_path / path).resolve()}')")
        arg_paths.append(path)
        chunks_len += len(arg_chunks[-1])
        if chunks_len + len(arg_chunks) >= limit:
            yield BATCH.format(",".join(arg_chunks[:-1])), arg_paths[:-1]
            arg_chunks = arg_chunks[-1:]
            arg_paths = arg_paths[-1:]
            chunks_len = len(arg_chunks[-1])
    yield BATCH.format(",".join(arg_chunks)), arg_paths

def _gen_rel_path_to_pyre_data_mapping(
    root_path: Path, paths: List[str], timeout: Optional[int]
) -> Mapping[str, PyreData]:
    result: dict[str, PyreData] = {}
    for batch_arg, batch_paths in _chunked_path_queries(root_path, paths):
        cmd_args = ["pyre", "--noninteractive", "query", batch_arg]
        try:
            stdout, stderr, return_code = _run_command(cmd_args, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise exc

        if return_code != 0:
            raise Exception(f"pyre exit code {return_code} stderr:\n {stderr}")
        try:
            batch_resp = json.loads(stdout)
            if "error" in batch_resp:
                print(f"Error in pyre query batch: {batch_resp['error']}")
                continue
            for path, item_resp in zip(batch_paths, batch_resp["response"]):
                if "error" in item_resp:
                    print(f"Error in pyre query types: {item_resp['error']}")
                    continue
                result.update({path: _process_pyre_data(data) for path, data in zip([path], item_resp["response"])})
        except Exception as e:
            raise Exception(f"{e}\n\nstderr:\n {stderr}")

    return result

def _run_command(
    cmd_args: List[str], timeout: Optional[int] = None
) -> Tuple[str, str, int]:
    process = subprocess.run(cmd_args, capture_output=True, timeout=timeout)
    return process.stdout.decode(), process.stderr.decode(), process.returncode


class RawPyreData(TypedDict):
    path: str
    types: Sequence[InferredType]


def _process_pyre_data(data: RawPyreData) -> PyreData:
    return {"types": sorted(data["types"], key=_sort_by_position)}


def _sort_by_position(data: InferredType) -> Tuple[int, int, int, int]:
    start = data["location"]["start"]
    stop = data["location"]["stop"]
    return start["line"], start["column"], stop["line"], stop["column"]

def _make_type_lookup(cache_types: Sequence[InferredType]) -> Dict[CodeRange, str]:
    lookup: Dict[CodeRange, str] = {}
    for item in cache_types:
        location = item["location"]
        start = location["start"]
        end = location["stop"]
        lookup[
            CodeRange(
                start=CodePosition(start["line"], start["column"]),
                end=CodePosition(end["line"], end["column"]),
            )
        ] = item["annotation"]
    return lookup
