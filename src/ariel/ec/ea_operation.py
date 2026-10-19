"""Operation protocol and the EAOperation step wrapper."""

import inspect
from collections.abc import Callable
from typing import Concatenate, ParamSpec, Protocol

from ariel.ec.population import Population

P = ParamSpec("P")


# -- Step wrapper --------------------------------------------------------------


class Operation(Protocol):
    """What the EA needs from an element of its pipeline.

    Every ``EAOperation``, whatever its bound arguments, fits this shape.
    """

    name: str

    def __call__(
        self, *args: object, **kwargs: object
    ) -> "Population | Operation":
        """Run on a population, or bind configuration arguments."""


class EAOperation[**P]:
    """EAOperation class.

    Decorating a function with ``@EAOperation`` validates its signature and
    turns it into a factory. Calling the decorated name with configuration
    arguments returns a new bound ``EAOperation`` instance ready to be placed
    in the ``ops`` list. The EA engine then calls each instance with a
    ``Population`` to execute it.

    Parameters
    ----------
    func : Callable[Concatenate[Population, P], Population]
        The pipeline function to wrap. Its first argument must be annotated
        as ``Population`` and its return type must be ``Population``.
    *args : P.args
        Positional configuration arguments bound at construction time.
    **kwargs : P.kwargs
        Keyword configuration arguments bound at construction time.

    Raises
    ------
    TypeError
        If the decorated function has no parameters.
    TypeError
        If the first parameter of the decorated function is not annotated
        as ``Population``.
    TypeError
        If the return annotation of the decorated function is not
        ``Population``.

    Examples
    --------
    >>> @EAOperation
    ... def mutate(
    ...     population: Population, probability: float = 0.2, span: int = 1
    ... ) -> Population:
    ...     ...
    ...     return population

    >>> ops: list[EAOperation] = [
    ...     mutate(probability=0.5, span=2),
    ...     parent_selection(tournament_size=5),
    ... ]
    """

    def __init__(
        self,
        func: Callable[Concatenate[Population, P], Population],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Wrap `func` with its bound positional and keyword arguments.

        Raises
        ------
        TypeError
            If an operation does not take and return a Population.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not args and not kwargs:
            if not params:
                msg = (
                    f"@EAOperation function '{func.__name__}' must have "
                    f"a Population as its first argument."
                )
                raise TypeError(
                    msg,
                )
            if params[0].annotation is not Population:
                annotation_name = getattr(
                    params[0].annotation,
                    "__name__",
                    params[0].annotation,
                )
                msg = (
                    f"@EAOperation function '{func.__name__}' first argument "
                    f"must be annotated as Population, got "
                    f"'{annotation_name}'."
                )
                raise TypeError(
                    msg,
                )
            if sig.return_annotation is not Population:
                return_name = getattr(
                    sig.return_annotation,
                    "__name__",
                    sig.return_annotation,
                )
                msg = (
                    f"@EAOperation function '{func.__name__}' must return "
                    f"Population, got "
                    f"'{return_name}'."
                )
                raise TypeError(
                    msg,
                )

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = func.__name__

    def __call__(
        self, *args: object, **kwargs: object
    ) -> "Population | EAOperation[P]":
        """
        Either bind configuration arguments or execute against a population.

        When the first positional argument is a ``Population`` the operator
        executes immediately and returns the result. Otherwise a new bound
        ``EAOperation`` instance is returned with the supplied arguments
        forwarded on every subsequent execution.

        Parameters
        ----------
        *args
            If the first element is a ``Population``, it is passed as the
            population argument to the wrapped function. Otherwise treated
            as configuration arguments to bind.
        **kwargs
            Configuration keyword arguments to bind.

        Returns
        -------
        Population
            When called with a ``Population`` as the first argument.
        EAOperation
            When called with configuration arguments, returning a new
            bound instance.

        Raises
        ------
        TypeError
            If the wrapped function does not return a ``Population``.
        """
        if args and isinstance(args[0], Population):
            population = args[0]
            result = self.func(population, *self.args, **self.kwargs)
            if not isinstance(result, Population):
                msg = (  # type: ignore[unreachable]  # known bug
                    f"'{self.func.__name__}' expected a Population as return "
                    f"value, got '{type(result).__name__}'."
                )
                raise TypeError(
                    msg,
                )
            return result

        return EAOperation(self.func, *args, **kwargs)  # type: ignore[arg-type]  # known bug

    def __repr__(self) -> str:
        """Return the function name with its bound keyword arguments.

        Returns
        -------
        str
            The function name and its bound keyword arguments.
        """
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())  # type: ignore[attr-defined]  # known bug
        return f"{self.func.__name__}({kwargs})"


# -- EA ------------------------------------------------------------------------
