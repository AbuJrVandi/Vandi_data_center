class DataAutomationError(Exception):
    """Base application error."""


class DataLoadError(DataAutomationError):
    """Raised when a dataset cannot be loaded."""


class DataGenerationError(DataAutomationError):
    """Raised when a dataset cannot be generated."""


class DataValidationError(DataAutomationError):
    """Raised when a requested data operation is invalid."""


class DataMergeError(DataAutomationError):
    """Raised when a merge request is invalid."""


class ExportError(DataAutomationError):
    """Raised when export cannot be completed."""
