from __future__ import annotations

import ast
import operator
from typing import Any

import pandas as pd

from .exceptions import DataValidationError
from .models import DatasetArtifact, OperationRecord


class SafeExpressionEvaluator(ast.NodeVisitor):
    BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
    CMP_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
    }

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def evaluate(self, expression: str) -> Any:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise DataValidationError(f"Invalid derived column expression: {expression}") from exc
        return self.visit(tree.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.dataframe.columns:
            raise DataValidationError(f"Unknown column in expression: {node.id}")
        return self.dataframe[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in self.BIN_OPS:
            raise DataValidationError("Expression contains an unsupported operator.")
        return self.BIN_OPS[op_type](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in self.UNARY_OPS:
            raise DataValidationError("Expression contains an unsupported unary operator.")
        return self.UNARY_OPS[op_type](self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = True
        for operation, comparator in zip(node.ops, node.comparators):
            op_type = type(operation)
            if op_type not in self.CMP_OPS:
                raise DataValidationError("Expression contains an unsupported comparison.")
            right = self.visit(comparator)
            result = self.CMP_OPS[op_type](left, right) & result
            left = right
        return result

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise DataValidationError("Unsupported function call in expression.")
        function_name = node.func.id
        args = [self.visit(argument) for argument in node.args]
        if function_name == "abs" and len(args) == 1:
            return abs(args[0])
        if function_name == "round" and len(args) in {1, 2}:
            digits = args[1] if len(args) == 2 else 0
            return args[0].round(digits) if hasattr(args[0], "round") else round(args[0], digits)
        raise DataValidationError(f"Unsupported function '{function_name}' in expression.")

    def generic_visit(self, node: ast.AST) -> Any:
        raise DataValidationError(f"Unsupported expression element: {type(node).__name__}")


class DataTransformer:
    def select_columns(self, dataset: DatasetArtifact, columns: list[str]) -> DatasetArtifact:
        self._validate_columns(dataset.dataframe, columns)
        result = dataset.clone(
            dataframe=dataset.dataframe.loc[:, columns].copy(deep=True),
            name=f"{dataset.name}_selected",
        )
        result.operation_history.append(
            OperationRecord(
                operation_name="select_columns",
                parameters={"columns": columns},
                summary=f"Selected {len(columns)} columns.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    def rename_columns(self, dataset: DatasetArtifact, rename_map: dict[str, str]) -> DatasetArtifact:
        self._validate_columns(dataset.dataframe, list(rename_map))
        dataframe = dataset.dataframe.rename(columns=rename_map).copy(deep=True)
        result = dataset.clone(dataframe=dataframe, name=f"{dataset.name}_renamed")
        result.operation_history.append(
            OperationRecord(
                operation_name="rename_columns",
                parameters={"rename_map": rename_map},
                summary=f"Renamed {len(rename_map)} columns.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    def derive_column(self, dataset: DatasetArtifact, *, new_column: str, expression: str) -> DatasetArtifact:
        dataframe = dataset.dataframe.copy(deep=True)
        evaluator = SafeExpressionEvaluator(dataframe)
        dataframe[new_column] = evaluator.evaluate(expression)
        result = dataset.clone(dataframe=dataframe, name=f"{dataset.name}_derived")
        result.operation_history.append(
            OperationRecord(
                operation_name="derive_column",
                parameters={"new_column": new_column, "expression": expression},
                summary=f"Derived column '{new_column}'.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    def convert_types(self, dataset: DatasetArtifact, conversions: dict[str, str]) -> DatasetArtifact:
        self._validate_columns(dataset.dataframe, list(conversions))
        dataframe = dataset.dataframe.copy(deep=True)
        for column, target_type in conversions.items():
            try:
                if target_type == "datetime64[ns]":
                    dataframe[column] = pd.to_datetime(dataframe[column], errors="raise")
                else:
                    dataframe[column] = dataframe[column].astype(target_type)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(f"Failed to convert column '{column}' to '{target_type}'.") from exc
        result = dataset.clone(dataframe=dataframe, name=f"{dataset.name}_typed")
        result.operation_history.append(
            OperationRecord(
                operation_name="convert_types",
                parameters={"conversions": conversions},
                summary=f"Converted {len(conversions)} columns.",
                dataset_before=dataset.name,
                dataset_after=result.name,
            )
        )
        return result

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
        missing_columns = sorted(set(columns) - set(dataframe.columns))
        if missing_columns:
            raise DataValidationError(f"Unknown columns: {', '.join(missing_columns)}")
