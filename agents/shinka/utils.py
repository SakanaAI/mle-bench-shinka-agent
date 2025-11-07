import signal
import time
from contextlib import contextmanager


class TrainingTimeoutError(TimeoutError):
    """Raised when `train_model` execution exceeds the allotted time."""


@contextmanager
def training_timeout(seconds: int):
    """Raise ``TrainingTimeoutError`` if the wrapped block exceeds ``seconds``."""

    if seconds <= 0:
        yield
        return

    # ``signal`` is only available on Unix platforms; fall back to manual check otherwise.
    if not hasattr(signal, "setitimer"):
        start = time.monotonic()
        yield
        if time.monotonic() - start > seconds:
            raise TrainingTimeoutError(
                f"train_model exceeded the allowed runtime of {seconds} seconds."
            )
        return

    def _handle_timeout(signum, frame):
        raise TrainingTimeoutError(
            f"train_model exceeded the allowed runtime of {seconds} seconds."
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
    except AttributeError:
        previous_timer = (0.0, 0.0)

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
