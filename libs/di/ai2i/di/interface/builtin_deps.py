from ai2i.di.interface.app_context import ApplicationContext
from ai2i.di.interface.models import RequestAndBody, RoundId, ScopedDependencyDefinition, TurnId

request = ScopedDependencyDefinition.predefined(RequestAndBody, "__internal_request_and_body__", "request")
turn_id = ScopedDependencyDefinition.predefined(TurnId, "__internal_turn_id__", "turn")
round_id = ScopedDependencyDefinition.predefined(RoundId, "__internal_round_id__", "round")
application_context = ScopedDependencyDefinition.predefined(ApplicationContext, "__internal_app_ctx__", "singleton")
