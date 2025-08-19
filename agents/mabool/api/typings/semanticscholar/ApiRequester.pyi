from _typeshed import Incomplete
from semanticscholar.SemanticScholarException import (
    BadQueryParametersException as BadQueryParametersException,
)
from semanticscholar.SemanticScholarException import (
    GatewayTimeoutException as GatewayTimeoutException,
)
from semanticscholar.SemanticScholarException import (
    InternalServerErrorException as InternalServerErrorException,
)
from semanticscholar.SemanticScholarException import (
    ObjectNotFoundException as ObjectNotFoundException,
)

logger: Incomplete

class ApiRequester:
    def __init__(self, timeout, retry: bool = True) -> None: ...
    @property
    def timeout(self) -> int: ...
    @timeout.setter
    def timeout(self, timeout: int) -> None: ...
    @property
    def retry(self) -> bool: ...
    @retry.setter
    def retry(self, retry: bool) -> None: ...
    async def get_data_async(
        self, url: str, parameters: str, headers: dict, payload: dict = None
    ) -> dict | list[dict]: ...
    def get_data(self, url: str, parameters: str, headers: dict, payload: dict = None) -> dict | list[dict]: ...
