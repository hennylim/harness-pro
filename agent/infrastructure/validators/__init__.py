"""
ValidatorRegistry – 언어 키 → IExecutionValidator 매핑 테이블.

새 언어 추가 시 이 딕셔너리에 항목 1개만 추가한다.
"""
from agent.domain.interfaces import IExecutionValidator
from agent.infrastructure.validators.python_validator import PythonExecutionValidator
from agent.infrastructure.validators.bash_validator import BashExecutionValidator
from agent.infrastructure.validators.c_cpp_validator import (
    CExecutionValidator,
    CppExecutionValidator,
)

VALIDATOR_REGISTRY: dict[str, IExecutionValidator] = {
    "python": PythonExecutionValidator(),
    "bash":   BashExecutionValidator(),
    "c":      CExecutionValidator(),
    "cpp":    CppExecutionValidator(),
}

__all__ = ["VALIDATOR_REGISTRY"]
