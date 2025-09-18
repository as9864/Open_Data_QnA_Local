# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .BuildSQLAgent import BuildSQLAgent
from .ValidateSQLAgent import ValidateSQLAgent
from .DebugSQLAgent import DebugSQLAgent
from .EmbedderAgent import EmbedderAgent
from .ResponseAgent import ResponseAgent
from .VisualizeAgent import VisualizeAgent
from .DescriptionAgent import DescriptionAgent

__all__ = [
    "BuildSQLAgent",
    "ValidateSQLAgent",
    "DebugSQLAgent",
    "EmbedderAgent",
    "ResponseAgent",
    "ResponseAgent_Local",
    "VisualizeAgent",
    "DescriptionAgent",
]


# def __getattr__(name):
#     if name == "BuildSQLAgent":
#         from .BuildSQLAgent import BuildSQLAgent
#
#         return BuildSQLAgent
#     if name == "ValidateSQLAgent":
#         from .ValidateSQLAgent import ValidateSQLAgent
#
#         return ValidateSQLAgent
#     if name == "DebugSQLAgent":
#         from .DebugSQLAgent import DebugSQLAgent
#
#         return DebugSQLAgent
#     if name == "EmbedderAgent":
#         from .EmbedderAgent import EmbedderAgent
#
#         return EmbedderAgent
#     if name == "ResponseAgent":
#         from .ResponseAgent import ResponseAgent
#
#         return ResponseAgent
#     if name == "ResponseAgent_Local":
#         from .ResponseAgent_Local import ResponseAgent as ResponseAgentLocal
#
#         return ResponseAgentLocal
#     if name == "VisualizeAgent":
#         from .VisualizeAgent import VisualizeAgent
#
#         return VisualizeAgent
#     if name == "DescriptionAgent":
#         from .DescriptionAgent import DescriptionAgent
#
#         return DescriptionAgent
#     raise AttributeError(name)
