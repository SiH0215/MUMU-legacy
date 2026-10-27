class MumuError(Exception):
    """사용자에게 원문 자격 증명 없이 표시할 수 있는 오류."""


class ConfigurationError(MumuError):
    pass


class InputError(MumuError):
    pass


class ServiceError(MumuError):
    pass
