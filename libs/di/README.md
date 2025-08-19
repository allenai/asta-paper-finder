# 🔌 ai2i.di

A dependency injection system for Python applications.

## Overview

The Dependency Injection system is intended to facilitate easier management of objects 
and their life cycles. 
- `Modules` are a hierarchical registry of providers
- `ApplicationContext` can be seen as the runtime instance of a particular module. It is responsible to 
initialize and orchestrate everything at run time. 

```python
from ai2i.di import create_module

sub_module = create_module("SubModule")

... 

top_module = create_module("TopModule", extends=[sub_module])

...

app_ctx = create_app_context(top_module, convtext_var)
```

## Main Concepts

### Providers 
Definitions of how to construct and release objects. Each object is defined using a single async factory
function or in an async context manager style (which allows the definition of a release logic)
Providers are associated with specific modules, so yo must create a module to register a provider to it.

```python
my_module = create_module("MyModule")

@my_module.provides(scope="singleton")
async def my_value() -> MyObject:
    return await MyObject.build()

@my_module.provides(scope="singleton")
async def my_value_with_destroy() -> AsyncIterator[MyObject]:
    obj = await MyObject.build()
    yield obj
    obj.release()
```

### Consumers
Regions of code that need objects defined by providers, the values can be accessed in two flavors:

1. Direct access through `get_dependency` that is available on the `DI` gateway object
```python
obj: MyObject = DI.get_dependency(my_value_with_destroy)
```
2. Managed decorator syntactic sugar (injecting values into function parameters)

```python
@DI.managed
async def compute(v: int, obj: MyObject = DI.requires(my_value)) -> int:
    return v + obj.value
```

The `managed` decorator syntax is the preferred way to consume values, as it helps in unit testing.
A managed function with a required parameter can easily be tested with a mock by simply providing a 
value for that parameter at the call site. 

__NOTE:__ `@DI.managed` supports decoration of functions, async functions, generators and async generators.

```python
@DI.managed
async def compute(v: int, obj: MyObject = DI.requires(my_value)) -> int:
    return v + obj.value

import unittest.mock.MagicMock


my_value_mock = MagicMock()
await compute(5, obj=my_value_mock)

assert my_value_mock ...
```

### Scopes
Scopes define the life cycle of the objects constructed by the providers.


##### Singleton
The singleton scope means that all objects will be constructed at the start of the application and
destroyed when the application ends

##### Request
The request scope means that all objects will be constructed at the beginning of
a request and released when the request ends. The `request` scope includes the sub_tasks lunched from 
the request, so the scope will close when the last sub_task finishes (if any).
In the request scope there is a special dependency
`builtin_deps.request` that can be required to produce a derivative object from the information contained in 
the request.

```python
from fastapi import Request
from ai2i.di import builtin_deps, RequestAndBody

@my_module.provides(scope="request")
async def request_user(rnb: RequestAndBody = DI.requires(builtin_deps.request)) -> User:
    return await User.from_id(rnb.request.headers["USER_ID"])
``` 

##### Turn
The turn scope means that all objects will be constructed at the beginning of
a single turn and released when the turn ends. The `turn` scope includes the sub_tasks lunched from 
within the turn, so the scope will close when the last sub_task finishes (if any).
In the turn scope there is a special dependency
`builtin_deps.turn_id` that can be required to produce a derivative objects from it

```python
from ai2i.di import builtin_deps

@my_module.provides(scope="turn")
async def turn_data(turn_id: TurnId = DI.requires(builtin_deps.turn_id)) -> TurnData:
    return await TurnData.from_id(turn_id)
``` 

##### Round
The round scope means that all objects will be constructed at the beginning of
a round and released when the round ends. In case the round doesn't end (no answer for clarification)
the round will be released after a timeout controlled by the config value `di.round_scope_timeout`
The `round` scope includes the sub_tasks lunched from within a round.
In the round scope there is a special dependency
`builtin_deps.round_id` that can be required to produce a derivative objects from it

```python
from ai2i.di import builtin_deps

@my_module.provides(scope="round")
async def round_data(round_id: RoundId = DI.requires(builtin_deps.round_id)) -> RoundData:
    return await Rounddata.from_id(round_id)
``` 

##  Additional Features

#### Transitivity of Providers
A `Provider` definition can also be a `Consumer` of a different `Provider`. The framework will automatically compute
the correct order in which to construct the objects defined by the providers. If it's impossible to find an order 
in which to construct the providers, a `CyclicDependencyError` will be raised. 

```python
@my_module.provides(scope="singleton")
async def db_engine() -> AsyncIterator[DBEngine]:
  engine = await DBEngine.build()
  yield engine
  engine.close()

@my_module.provides(scope="request")
async def transaction(engine: DBEngine = DI.require(db_engine)) -> AsyncIterator[Transaction]:
  t = db_engine.transaction()
  yield t
  t.close()

@DI.managed
async def do_operation(transaction: Transaction = DI.require(transactions)) -> None:
    with transaction.start():
        ...
```


#### Config Integration
We often want to control the construction of objects using configuration.
Functions decorated with `@X.provides` and `@DI.managed` also support configuration injection
through the `DI.config` definition. (this is just an alias for `ConfigValue` and is 
intended to be a variant that closely resemble the `DI.require` syntax)

```python
@my_module.provides(scope="singleton")
async def storage_service(db_url: str = DI.config(cfg_schema.db.url)) -> StorageService:
  return await DBStorage.build(db_url)

@DI.managed
async def fetch_documents(
  limit: int = DI.config(cfg_schema.results.limit),
  storage_service: StorageService = DI.requires(storage_service)
) -> int:
  return storage_service.fetch_documents(limit=limit)
```

#### Manually Naming Providers
In rare cases we might want to manually assign the name of the provider. At the moment the only use case for 
this is to fix issues with cyclic imports. In an ideally organized system this shouldn't be required at all.

It's possible to define the name in the `@provide` decorator parameters and use the matching 
`app_ctx.get_depdendency_by_name` method.

```python
@my_module.provides(scope="singleton", name="my_message")
async def greeting_message() -> str:
  return "Hello There"

...

#in some other place inside a "singleton" scope
message = DI.get_depdendency_by_name("my_message")
```


#### Testing with Custom Scope
Sometimes we need to test code that needs dependencies, but the required dependencies can not be overridden 
by simply passing a parameter to a function (Like the mocking example in the `Consumers` section).

For this purpose there is a special `custom` scope on `app_ctx` that can be manually initialized with values for 
the different requirements of the code you are testing. 

```python
@my_module.provides(scope="singleton")
async def greeting_message() -> str:
  return "Hello There"


app_ctx = create_app_context(my_module, ctxvar)

async with app_ctx.scopes.custom.managed_scope({
    "manually_named_printer": printer, 
    greeting_message.uniqe_name: "Sup"
}):
    # Code that internally will use the printer to print a greeting message
```

## Development

### Setup Development Environment

```shell script
make sync-dev
```

### Testing

```shell script
make test
# -or-
make test-cov  # With coverage
```

### Code Quality

```shell script
make style  # Run format check, lint and type-check
make fix    # Automatically fix style issues
```

## Misc

##### Scopes illustration

![Illustration of scope lifetimes](./.readme_assets/scopes_illustration.svg)
