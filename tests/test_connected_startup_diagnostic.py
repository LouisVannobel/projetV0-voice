"""Tiny offline checks: load only the test scenario's two stdlib observers."""

import ast
import asyncio
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


class ConnectedStartupDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).parent / "integration" / "sparra_connected_scenario.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"observe_native_startup", "safe_startup_diagnostic"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name in names]
        if {node.name for node in definitions} != names:
            raise AssertionError("missing_test_side_startup_observer")
        namespace = {"asyncio": asyncio, "patch": patch}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
             namespace)
        cls.observe = staticmethod(namespace["observe_native_startup"])
        cls.safe = staticmethod(namespace["safe_startup_diagnostic"])

    def test_terminal_codes_are_exact_and_closed(self):
        for code in ("runtime_startup_invalid", "stale_recovery_failed",
                     "qualification_run_consumed", "writer_startup_failed",
                     "writer_quick_check_failed", "owned_task_registration_failed",
                     "runtime_startup_failed"):
            with self.subTest(code=code):
                self.assertEqual(self.safe(RuntimeError(code), {}, "backlog")["terminal_code"],
                                 code)

    def test_unknown_payload_and_class_are_redacted_without_stringifying(self):
        secret = "synthetic-secret-sentinel/provider/request/source-key"

        class ProviderSecretError(RuntimeError):
            def __str__(self):
                raise AssertionError("must_not_stringify")

        for error in (RuntimeError(secret), RuntimeError("stale_recovery_failed", secret),
                      AssertionError("native_" + secret), ProviderSecretError(secret),
                      RuntimeError([secret])):
            value = self.safe(error, {"last_phase": secret, "outcome": secret}, secret)
            self.assertNotIn(secret, json.dumps(value))
            self.assertEqual(value["terminal_code"], "other_safe_failure")
            self.assertEqual(value["last_phase"], "other_safe_failure")
            self.assertEqual(value["outcome"], "other")
        self.assertEqual(self.safe(ProviderSecretError(secret), {}, None)["error_class"],
                         "OtherException")

    def test_str_subclasses_do_not_enter_allowlists(self):
        class PretendCode(str):
            pass

        value = self.safe(RuntimeError(PretendCode("stale_recovery_failed")),
                          {"last_phase": PretendCode("stale_recovery_failed")},
                          PretendCode("backlog"))
        self.assertEqual(value["terminal_code"], "other_safe_failure")
        self.assertEqual(value["last_phase"], "other_safe_failure")
        self.assertEqual(value["recovery_case"], "other")

    def test_frames_exclude_paths_arbitrary_names_and_are_bounded(self):
        def provider_secret_source_key():
            raise RuntimeError("synthetic-secret-sentinel")

        def setup():
            provider_secret_source_key()

        try:
            setup()
        except RuntimeError as error:
            value = self.safe(error, {}, "held")
        self.assertEqual(value["frames"], ["setup"])
        self.assertNotIn("provider_secret_source_key", json.dumps(value))
        self.assertNotIn(str(Path(__file__)), json.dumps(value))

        def startup(depth):
            if depth:
                startup(depth - 1)
            else:
                raise RuntimeError("runtime_startup_failed")

        try:
            startup(20)
        except RuntimeError as error:
            self.assertEqual(self.safe(error, {}, None)["frames"], ["startup"] * 8)

    def test_recovery_cases_are_fixed(self):
        for case in ("held", "backlog", "final-fix", "replay"):
            self.assertEqual(self.safe(RuntimeError(), {}, case)["recovery_case"], case)
        self.assertEqual(self.safe(RuntimeError(), {}, None)["recovery_case"], "other")

    async def exercise(self, error=None, timed_out=False, code="stale_recovery_failed"):
        calls = []
        result = object()
        awaitable = asyncio.get_running_loop().create_future()
        awaitable.set_result(result)

        class Supervisor:
            _startup_phase_timed_out = timed_out
            _startup_phase_timeout_seconds = 5.0
            _startup_phase_timeouts = {"stale_recovery_failed": 5.0}

            async def _startup_await(self, argument, *, code):
                calls.append((argument, code))
                if error is not None:
                    raise error
                return await argument

        supervisor = Supervisor()
        original = supervisor._startup_await
        diagnostic = {}
        with ExitStack() as stack:
            self.observe(supervisor, stack, diagnostic)
            if error is None:
                self.assertIs(await supervisor._startup_await(awaitable, code=code), result)
            else:
                try:
                    await supervisor._startup_await(awaitable, code=code)
                except BaseException as caught:
                    self.assertIs(caught, error)
                else:
                    self.fail("original_exception_not_preserved")
        self.assertEqual(calls, [(awaitable, code)])
        self.assertEqual(supervisor._startup_await, original)
        self.assertNotIn("_startup_await", supervisor.__dict__)
        self.assertEqual(supervisor._startup_phase_timeout_seconds, 5.0)
        self.assertEqual(supervisor._startup_phase_timeouts, {"stale_recovery_failed": 5.0})
        return diagnostic

    async def test_success_preserves_original_awaitable_result_and_single_call(self):
        self.assertEqual(await self.exercise(),
                         {"last_phase": "stale_recovery_failed", "outcome": "other"})

    async def test_failure_preserves_exception_identity_and_native_timeout_flag(self):
        for timed_out, outcome in ((True, "native_timeout"), (False, "native_exception"),
                                  ("synthetic-secret-sentinel", "native_exception")):
            with self.subTest(timed_out=timed_out):
                value = await self.exercise(RuntimeError("stale_recovery_failed"), timed_out)
                self.assertEqual(value["outcome"], outcome)

    async def test_cancellation_is_rethrown_unchanged(self):
        value = await self.exercise(asyncio.CancelledError())
        self.assertEqual(value["outcome"], "native_exception")

    async def test_unknown_phase_is_redacted_but_forwarded_unchanged(self):
        value = await self.exercise(RuntimeError("synthetic-secret-sentinel"),
                                    code="synthetic-secret-sentinel")
        self.assertEqual(value["last_phase"], "other_safe_failure")
        self.assertNotIn("synthetic-secret-sentinel", json.dumps(value))


if __name__ == "__main__":
    unittest.main()
