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


def __getattr__(name):
    if name == "BuildSQLAgent":
        from .BuildSQLAgent import BuildSQLAgent

        return BuildSQLAgent
    if name == "ValidateSQLAgent":
        from .ValidateSQLAgent import ValidateSQLAgent

        return ValidateSQLAgent
    if name == "DebugSQLAgent":
        from .DebugSQLAgent import DebugSQLAgent

        return DebugSQLAgent
    if name == "EmbedderAgent":
        from .EmbedderAgent import EmbedderAgent

        return EmbedderAgent
    if name == "ResponseAgent":
        from .ResponseAgent import ResponseAgent

        return ResponseAgent
    if name == "ResponseAgent_Local":
        from .ResponseAgent_Local import ResponseAgent as ResponseAgentLocal

        return ResponseAgentLocal
    if name == "VisualizeAgent":
        from .VisualizeAgent import VisualizeAgent

        return VisualizeAgent
    if name == "DescriptionAgent":
        from .DescriptionAgent import DescriptionAgent

        return DescriptionAgent
    raise AttributeError(name)
