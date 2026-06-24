"""
Language profile registry.

``PROFILE_REGISTRY`` maps TargetLanguage enum value (str) → ILanguageProfile instance.
언어를 추가할 때 이 딕셔너리에만 항목을 추가하면 됩니다.
"""
from agent.domain.profiles.python_profile import PythonProfile
from agent.domain.profiles.bash_profile import BashProfile
from agent.domain.profiles.c_profile import CProfile
from agent.domain.profiles.cpp_profile import CppProfile
from agent.domain.language_profile import ILanguageProfile

PROFILE_REGISTRY: dict[str, ILanguageProfile] = {
    "python": PythonProfile(),
    "bash":   BashProfile(),
    "c":      CProfile(),
    "cpp":    CppProfile(),
}

__all__ = [
    "ILanguageProfile",
    "PROFILE_REGISTRY",
    "PythonProfile",
    "BashProfile",
    "CProfile",
    "CppProfile",
]
