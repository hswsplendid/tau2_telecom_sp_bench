"""tau2 telecom agent with compression, ABC logging, timing, and checkpoints."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

from pydantic import Field

import sp_config as CFG

for import_path in (CFG.TAU2_SRC, CFG.COMPRESSOR_ROOT, CFG.BENCH_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from compressor import ContextCompressor, estimate_tokens, should_compact  # noqa: E402
from tau2.agent.base_agent import ValidAgentInputMessage  # noqa: E402
from tau2.agent.llm_agent import LLMAgent, LLMAgentState  # noqa: E402
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, UserMessage  # noqa: E402
from tau2.environment.tool import Tool  # noqa: E402
from tau2.utils.llm_utils import generate, to_litellm_messages, to_tau2_messages  # noqa: E402


def safe_sample_id(task_id: str) -> str:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:10]
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in task_id[:80]).strip("_")
    return f"{cleaned}_{digest}" if cleaned else digest


def message_token_rows(messages: list[dict]) -> list[dict]:
    rows = []
    for index, message in enumerate(messages):
        rows.append(
            {
                "index": index,
                "role": message.get("role"),
                "tokens": estimate_tokens([message]),
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
                "tool_call_id": message.get("tool_call_id"),
            }
        )
    return rows


CUSTOMER_DEVICE_TOOL_NAMES = {
    "can_send_mms",
    "check_apn_settings",
    "check_app_permissions",
    "check_data_restriction_status",
    "check_network_mode_preference",
    "check_network_status",
    "check_sim_status",
    "check_status_bar",
    "check_vpn_status",
    "check_wifi_calling_status",
    "disconnect_vpn",
    "grant_app_permission",
    "reboot_device",
    "reseat_sim_card",
    "reset_apn_settings",
    "run_speed_test",
    "set_network_mode_preference",
    "toggle_airplane_mode",
    "toggle_data",
    "toggle_data_saver_mode",
    "toggle_roaming",
    "toggle_wifi_calling",
}


class TelecomSPAgentState(LLMAgentState):
    previous_summary: Optional[str] = None
    turn_count: int = 0
    step_count: int = 0
    compression_count: int = 0
    tool_call_count: int = 0
    prompt_entries: list = Field(default_factory=list)
    timing_log: list = Field(default_factory=list)
    abc_snapshots: list = Field(default_factory=list)
    turn_traces: list = Field(default_factory=list)
    checkpoints: list = Field(default_factory=list)
    reference_context_tokens: int = 0
    last_was_compression: bool = False


class TelecomSPAgent(LLMAgent):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        compress_enabled: bool,
        model_key: str,
        task_id: str,
        run_root: str | Path,
        mode_label: str,
        initial_context_mode: str,
        target_initial_tokens: int,
        context_window: int,
        reserve_tokens: int,
        keep_recent_tokens: int,
        summary_max_tokens: int,
        include_task_ticket: bool,
        task_ticket: str | None,
        stepwise_tech_support: bool,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.compress_enabled = compress_enabled
        self.model_key = model_key
        self.task_id = task_id
        self.include_task_ticket = include_task_ticket
        self.task_ticket = task_ticket
        self.stepwise_tech_support = stepwise_tech_support
        self.sample_id = safe_sample_id(task_id or "unknown_task")
        self.mode_label = mode_label
        self.initial_context_mode = initial_context_mode
        self.target_initial_tokens = target_initial_tokens
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.summary_max_tokens = summary_max_tokens
        self.available_tool_names = {
            name
            for tool in tools
            for name in [getattr(tool, "name", None), getattr(getattr(tool, "function", None), "name", None)]
            if name
        }

        self.run_root = Path(run_root)
        self.prompt_log_dir = self.run_root / "prompt_logs"
        self.trace_dir = self.run_root / "traces"
        self.abc_dir = self.run_root / "abc_segments"
        self.timing_dir = self.run_root / "timing"
        self.checkpoint_dir = self.run_root / "checkpoints"
        for directory in [self.prompt_log_dir, self.trace_dir, self.abc_dir, self.timing_dir, self.checkpoint_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.compressor = None
        if compress_enabled:
            model_path = CFG.MODEL_REGISTRY.get(model_key, {}).get("model_path", model_key)
            self.compressor = ContextCompressor(
                api_base=CFG.PROXY_URL,
                model_name=model_path,
                api_key=CFG.API_KEY,
                summary_max_tokens=summary_max_tokens,
                context_window=context_window,
                reserve_tokens=reserve_tokens,
                keep_recent_tokens=keep_recent_tokens,
                quality_guard_enabled=CFG.QUALITY_GUARD_ENABLED,
                quality_guard_max_retries=CFG.QUALITY_GUARD_MAX_RETRIES,
                use_structured_instructions=CFG.USE_STRUCTURED_INSTRUCTIONS,
                preserved_recent_turns=CFG.PRESERVED_RECENT_TURNS,
            )

    @property
    def system_prompt(self) -> str:
        prompt = super().system_prompt
        if getattr(self, "stepwise_tech_support", False):
            prompt = (
                f"{prompt}\n"
                "<workflow_guidance>\n"
                "When facts can be checked with available telecom tools, make the tool call instead of telling the user you will check it. "
                "Do not say that you are checking account, line, usage, roaming, billing, or device state unless you are making the corresponding tool call in that turn.\n"
                "Never invent tool names. Only call tools that are present in the provided tool schema. In particular, do not call get_lines_for_customer; after get_customer_by_phone returns line_ids, inspect candidate lines with get_details_by_id using those IDs until the phone_number matches the ticket or user phone number.\n"
                "Customer-device troubleshooting actions and diagnostics are not agent tools. Never call or output customer-side function names as tool calls, including check_network_status, check_status_bar, check_sim_status, check_network_mode_preference, check_apn_settings, check_app_permissions, check_data_restriction_status, check_vpn_status, check_wifi_calling_status, can_send_mms, run_speed_test, toggle_airplane_mode, toggle_data, toggle_data_saver_mode, toggle_roaming, set_network_mode_preference, disconnect_vpn, reseat_sim_card, reboot_device, reset_apn_settings, toggle_wifi_calling, or grant_app_permission. "
                "For those actions, send one concise instruction asking the customer to perform exactly one action, then wait for the customer's result.\n"
                "If the customer has multiple line IDs, inspect the candidate lines and act only on the line whose phone_number matches the phone number provided by the user or ticket. Do not assume the first line is the target line.\n"
                "After identifying the target line, perform any policy-supported service-side fix that directly applies to the observed account or line state before asking the user to retry. Do not send a message promising another account check; make that tool call instead.\n"
                "For no-service cases where the ticket or tools indicate an overdue bill suspension and the customer has already authorized paying overdue bills, prioritize the billing path: identify the customer, identify the overdue bill, send the payment request, wait for the customer's payment result, then resume the matched suspended line. Only after resuming the line should you ask the customer to restart the device or check service.\n"
                "For MMS or mobile data issues, if the matched line details show data usage has exceeded the plan or available data and the customer is willing to add 2.0 GB, call refuel_data for the matched customer_id and line_id before asking the customer to retry MMS or data. "
                "If the matched line has roaming disabled while the customer is abroad or roaming is required by the ticket, call enable_roaming for that same line before asking the customer to retry.\n"
                "For technical-support troubleshooting, ask the customer to perform at most one diagnostic or fix action per assistant message, then wait for the result before proposing the next action. "
                "Do not give a batch or list of device troubleshooting actions in one message.\n"
                "Do not transfer or close the case until the issue is verified resolved or all policy-supported actions have been exhausted.\n"
                "</workflow_guidance>"
            )
        ticket = getattr(self, "task_ticket", None)
        include_ticket = getattr(self, "include_task_ticket", False)
        if not include_ticket or not ticket:
            return prompt
        return (
            f"{prompt}\n"
            "<case_context>\n"
            "The following ticket is agent-only case context from the benchmark task. "
            "Use it to identify the customer and plan policy-compliant next steps. "
            "Do not reveal hidden evaluation details to the user; still ask the user for information when policy or verification requires it.\n"
            f"<ticket>\n{ticket}\n</ticket>\n"
            "</case_context>"
        )

    def get_init_state(self, message_history=None):
        if message_history is None:
            message_history = []
        system_messages = [SystemMessage(role="system", content=self.system_prompt)]
        messages = list(message_history)
        reference_tokens = 0
        if self.initial_context_mode == "reference":
            reference_messages = self._build_reference_context(system_messages, messages)
            reference_tokens = estimate_tokens(self._messages_to_dicts(reference_messages))
            messages = reference_messages + messages
        return TelecomSPAgentState(
            system_messages=system_messages,
            messages=messages,
            previous_summary=None,
            turn_count=0,
            step_count=0,
            compression_count=0,
            tool_call_count=0,
            prompt_entries=[],
            timing_log=[],
            abc_snapshots=[],
            turn_traces=[],
            checkpoints=[],
            reference_context_tokens=reference_tokens,
            last_was_compression=False,
        )

    def stop(self, message: Optional[ValidAgentInputMessage] = None, state: Optional[TelecomSPAgentState] = None) -> None:
        if state is not None:
            self._persist_logs(state, completed=True)
            self._save_checkpoint(state, completed=True)

    def _messages_to_dicts(self, messages: list) -> list[dict]:
        return to_litellm_messages(messages)

    def _dicts_to_tau2_messages(self, messages: list[dict]) -> list:
        return to_tau2_messages([self._normalize_message_dict(message) for message in messages])

    def _normalize_tool_call_dict(self, tool_call: dict) -> dict:
        function = tool_call.get("function") or {}
        arguments = tool_call.get("arguments", function.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return {
            "id": tool_call.get("id", ""),
            "name": tool_call.get("name") or function.get("name"),
            "arguments": arguments or {},
            "requestor": tool_call.get("requestor", "assistant"),
        }

    def _normalize_message_dict(self, message: dict) -> dict:
        normalized = dict(message)
        if normalized.get("role") == "tool" and not normalized.get("id"):
            normalized["id"] = normalized.get("tool_call_id", "")
        if normalized.get("tool_calls"):
            normalized["tool_calls"] = [self._normalize_tool_call_dict(tool_call) for tool_call in normalized["tool_calls"]]
        return normalized

    def _blocked_tool_response(self, tool_name: str, arguments: dict) -> str:
        customer_instructions = {
            "can_send_mms": "Please try sending an MMS message now and tell me whether it goes through.",
            "check_apn_settings": "Please check your APN and picture messaging settings and tell me what you see.",
            "check_app_permissions": "Please check whether the messaging app has the required permissions and tell me the result.",
            "check_data_restriction_status": "Please check whether mobile data is restricted for the messaging app and tell me the result.",
            "check_network_mode_preference": "Please check your preferred network mode and tell me what it is set to.",
            "check_network_status": "Please check your cellular connection status and tell me what it shows.",
            "check_sim_status": "Please check whether your SIM is active and tell me the result.",
            "check_status_bar": "Please look at your status bar and tell me the signal, network type, and mobile data indicators.",
            "check_vpn_status": "Please check whether a VPN is connected and tell me the result.",
            "check_wifi_calling_status": "Please check whether Wi-Fi calling is enabled and tell me the result.",
            "disconnect_vpn": "Please disconnect your VPN, then tell me when that is done.",
            "grant_app_permission": "Please grant the messaging app the requested permission, then tell me when that is done.",
            "reboot_device": "Please restart your device, then tell me when it is back on.",
            "reseat_sim_card": "Please reseat your SIM card, then tell me when that is done.",
            "reset_apn_settings": "Please reset your APN settings, then tell me when that is done.",
            "run_speed_test": "Please run a speed test and tell me the result.",
            "set_network_mode_preference": "Please set your preferred network mode to 4G or 5G preferred, then tell me when that is done.",
            "toggle_airplane_mode": "Please turn airplane mode off, then tell me when that is done.",
            "toggle_data": "Please turn mobile data on, then tell me when that is done.",
            "toggle_data_saver_mode": "Please turn data saver off, then tell me when that is done.",
            "toggle_roaming": "Please turn data roaming on, then tell me when that is done.",
            "toggle_wifi_calling": "Please turn Wi-Fi calling off, then tell me when that is done.",
        }
        if tool_name in customer_instructions:
            return customer_instructions[tool_name]
        if tool_name == "get_lines_for_customer":
            return "I found multiple lines on the account. I will check the listed line IDs one at a time to match your phone number."
        return "I need to continue using the available account tools and one customer step at a time."

    def _text_tool_call_dict(self, content: str | None) -> dict | None:
        candidate = (content or "").strip()
        if not candidate:
            return None
        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate.strip("`").strip()
        try:
            parsed = ast.parse(candidate, mode="eval")
        except SyntaxError:
            return None
        call = parsed.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
        tool_name = call.func.id
        arguments = {}
        if len(call.args) == 1 and isinstance(call.args[0], ast.Dict):
            try:
                arguments.update(ast.literal_eval(call.args[0]))
            except (ValueError, SyntaxError):
                return None
        elif call.args:
            return None
        for keyword in call.keywords:
            if keyword.arg is None:
                return None
            try:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                return None
        digest = hashlib.sha1(f"{candidate}:{time.time()}".encode("utf-8")).hexdigest()[:24]
        return {"id": f"call_{digest}", "name": tool_name, "arguments": arguments, "requestor": "assistant"}

    def _normalize_assistant_message(self, message: AssistantMessage) -> AssistantMessage:
        if not message.tool_calls:
            text_tool_call = self._text_tool_call_dict(message.content)
            if text_tool_call is None:
                return message
            if text_tool_call["name"] in CUSTOMER_DEVICE_TOOL_NAMES or (
                self.available_tool_names and text_tool_call["name"] not in self.available_tool_names
            ):
                return AssistantMessage(
                    role="assistant",
                    content=self._blocked_tool_response(text_tool_call["name"], text_tool_call["arguments"]),
                    is_audio=message.is_audio,
                    turn_idx=message.turn_idx,
                    cost=message.cost,
                    usage=message.usage,
                    raw_data=message.raw_data,
                    generation_time_seconds=message.generation_time_seconds,
                )
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(**text_tool_call)],
                is_audio=message.is_audio,
                turn_idx=message.turn_idx,
                cost=message.cost,
                usage=message.usage,
                raw_data=message.raw_data,
                generation_time_seconds=message.generation_time_seconds,
            )
        normalized_tool_calls = [self._normalize_tool_call_dict(tool_call.model_dump()) for tool_call in message.tool_calls]
        blocked_tool_call = next(
            (
                tool_call
                for tool_call in normalized_tool_calls
                if tool_call["name"] in CUSTOMER_DEVICE_TOOL_NAMES
                or (self.available_tool_names and tool_call["name"] not in self.available_tool_names)
            ),
            None,
        )
        if blocked_tool_call is not None:
            return AssistantMessage(
                role="assistant",
                content=self._blocked_tool_response(blocked_tool_call["name"], blocked_tool_call["arguments"]),
                is_audio=message.is_audio,
                turn_idx=message.turn_idx,
                cost=message.cost,
                usage=message.usage,
                raw_data=message.raw_data,
                generation_time_seconds=message.generation_time_seconds,
            )
        return AssistantMessage(
            role="assistant",
            content=message.content,
            tool_calls=[ToolCall(**tool_call) for tool_call in normalized_tool_calls],
            is_audio=message.is_audio,
            turn_idx=message.turn_idx,
            cost=message.cost,
            usage=message.usage,
            raw_data=message.raw_data,
            generation_time_seconds=message.generation_time_seconds,
        )

    def _build_reference_context(self, system_messages: list, message_history: list) -> list:
        seed = (
            "This benchmark-only reference context summarizes generic telecom troubleshooting, "
            "billing, line management, roaming, data usage, payment, SIM, APN, Wi-Fi calling, "
            "VPN, network mode, device restart, and escalation procedures. It is not a customer "
            "request and contains no task-specific answer. The live customer request appears later."
        )
        blocks = []
        block_index = 0
        current_messages = system_messages + message_history
        while estimate_tokens(self._messages_to_dicts(current_messages + blocks)) < self.target_initial_tokens:
            detail_lines = []
            for item_index in range(12):
                serial = block_index * 12 + item_index + 1
                detail_lines.append(
                    f"Case pattern {serial}: verify identity, inspect account line state, compare plan limits, "
                    f"confirm device-side setting {serial % 7}, run a reversible diagnostic, document the result, "
                    f"and avoid irreversible changes unless policy and customer confirmation allow it."
                )
            content = seed + "\n" + "\n".join(detail_lines)
            blocks.extend(
                self._dicts_to_tau2_messages(
                    [
                        {"role": "user", "content": f"[Benchmark reference context block {block_index + 1}]\n{content}"},
                        {"role": "assistant", "content": "Acknowledged. I will treat this as background reference only and follow the actual customer request when it arrives."},
                    ]
                )
            )
            block_index += 1
            if block_index > 80:
                break
        return blocks

    def _count_user_turns(self, messages: list) -> int:
        return sum(1 for message in messages if isinstance(message, UserMessage))

    def _extract_b2_c1_after(self, compressed_messages: list[dict]) -> tuple[list[dict], list[dict]]:
        rest = compressed_messages[1:] if compressed_messages and compressed_messages[0].get("role") == "system" else list(compressed_messages)
        b2_messages = []
        index = 0
        while index < len(rest):
            content = str(rest[index].get("content") or "")
            role = rest[index].get("role")
            is_summary_user = role == "user" and ("Previous conversation summary" in content or "conversation summary" in content.lower())
            is_summary_ack = role == "assistant" and "previous conversation" in content.lower()
            if is_summary_user or (b2_messages and is_summary_ack):
                b2_messages.append(rest[index])
                index += 1
                continue
            break
        return b2_messages, rest[index:]

    def _build_abc_snapshot(self, before: list[dict], after: list[dict], comp_info: dict, event_base: dict) -> dict:
        has_system = bool(before and before[0].get("role") == "system")
        prefix_end = 1 if has_system else 0
        cut_index = comp_info.get("cut_index", prefix_end)
        before_a = before[:prefix_end]
        before_b1 = before[prefix_end:cut_index]
        before_c1 = before[cut_index:]

        after_a = after[:prefix_end] if has_system else []
        after_b2, after_c1 = self._extract_b2_c1_after(after)
        if before_c1 and len(before_c1) != len(after_c1):
            before_c1 = before[-len(after_c1):] if after_c1 else []
            before_b1 = before[prefix_end : len(before) - len(before_c1)]

        before_b1_tokens = estimate_tokens(before_b1)
        after_b2_tokens = estimate_tokens(after_b2)
        after_c1_tokens = estimate_tokens(after_c1)
        return {
            **event_base,
            "abc_segments": {
                "before": {
                    "A": before_a,
                    "B1": before_b1,
                    "C1": before_c1,
                    "A_tokens": estimate_tokens(before_a),
                    "B1_tokens": before_b1_tokens,
                    "C1_tokens": estimate_tokens(before_c1),
                    "message_tokens": {
                        "A": message_token_rows(before_a),
                        "B1": message_token_rows(before_b1),
                        "C1": message_token_rows(before_c1),
                    },
                },
                "after": {
                    "A": after_a,
                    "B2": after_b2,
                    "C1": after_c1,
                    "A_tokens": estimate_tokens(after_a),
                    "B2_tokens": after_b2_tokens,
                    "C1_tokens": after_c1_tokens,
                    "B2_plus_C1_tokens": after_b2_tokens + after_c1_tokens,
                    "message_tokens": {
                        "A": message_token_rows(after_a),
                        "B2": message_token_rows(after_b2),
                        "C1": message_token_rows(after_c1),
                    },
                },
            },
            "B1_tokens": before_b1_tokens,
            "B2_tokens": after_b2_tokens,
            "B2_to_B1_ratio": round(after_b2_tokens / max(before_b1_tokens, 1), 6),
            "C1_tokens_after": after_c1_tokens,
            "B2_plus_C1_tokens": after_b2_tokens + after_c1_tokens,
            "C1_valid_min_2000": after_c1_tokens >= CFG.C1_MIN_TOKENS,
            "C1_gt_3000_diagnostic": after_c1_tokens > CFG.C1_DIAGNOSTIC_MAX_TOKENS,
            "compression_info": comp_info,
        }

    def _maybe_compress(self, state: TelecomSPAgentState) -> dict | None:
        if not self.compress_enabled or self.compressor is None:
            return None
        if state.step_count < 5:
            return None
        all_messages = state.system_messages + state.messages
        message_dicts = self._messages_to_dicts(all_messages)
        total_tokens = estimate_tokens(message_dicts)
        if not should_compact(total_tokens, self.context_window, self.reserve_tokens):
            return None

        started = time.perf_counter()
        before = json.loads(json.dumps(message_dicts))
        compressed_messages, comp_info = self.compressor.compress(
            message_dicts,
            keep_recent_turns=CFG.KEEP_RECENT_TURNS,
            previous_summary=state.previous_summary,
            use_token_budget=True,
        )
        elapsed_s = time.perf_counter() - started
        if comp_info is None:
            return None

        after = json.loads(json.dumps(compressed_messages))
        state.previous_summary = comp_info.get("summary_text", state.previous_summary)
        state.compression_count += 1
        post_tokens = estimate_tokens(after)
        event_base = {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "turn": state.turn_count,
            "step": state.step_count,
            "compression_index": state.compression_count,
            "pre_prompt_tokens": total_tokens,
            "post_prompt_tokens": post_tokens,
            "token_saving_pct": round((1 - post_tokens / max(total_tokens, 1)) * 100, 3),
            "summary_generation_time_s": round(elapsed_s, 4),
            "threshold_tokens": self.context_window - self.reserve_tokens,
            "context_window": self.context_window,
            "reserve_tokens": self.reserve_tokens,
            "keep_recent_tokens": self.keep_recent_tokens,
        }
        snapshot = self._build_abc_snapshot(before, after, comp_info, event_base)
        state.abc_snapshots.append(snapshot)
        state.messages = [message for message in self._dicts_to_tau2_messages(compressed_messages) if not isinstance(message, SystemMessage)]
        state.system_messages = [message for message in self._dicts_to_tau2_messages(compressed_messages) if isinstance(message, SystemMessage)] or state.system_messages
        state.last_was_compression = True
        return snapshot

    def _generate_next_message(self, message: ValidAgentInputMessage, state: TelecomSPAgentState) -> AssistantMessage:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        if isinstance(message, UserMessage):
            state.turn_count += 1

        compression_event = self._maybe_compress(state)
        messages = state.system_messages + state.messages
        prompt_dicts = self._messages_to_dicts(messages)
        prompt_tokens_est = estimate_tokens(prompt_dicts)
        state.step_count += 1

        classification = "incremental"
        if len(state.timing_log) == 0:
            classification = "full_prefill"
        elif compression_event is not None:
            classification = "semi_prefill"

        started = time.perf_counter()
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        assistant_message = self._normalize_assistant_message(assistant_message)
        elapsed_s = time.perf_counter() - started

        usage = assistant_message.usage or {}
        output_tokens = usage.get("completion_tokens") or estimate_tokens(
            [{"role": "assistant", "content": assistant_message.content or ""}]
        )
        tool_calls = [tool_call.model_dump() for tool_call in assistant_message.tool_calls] if assistant_message.tool_calls else []
        state.tool_call_count += len(tool_calls)

        timing_entry = {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "turn": state.turn_count,
            "step": state.step_count,
            "classification": classification,
            "prompt_tokens": prompt_tokens_est,
            "output_tokens": output_tokens,
            "total_ms": round(elapsed_s * 1000, 3),
            "ttft_ms": None,
            "decode_ms": None,
            "timing_note": "litellm non-streaming total latency; TTFT/decode split requires streaming or server metrics",
            "compressed_this_step": compression_event is not None,
            "semi_prefill_tokens": compression_event.get("B2_plus_C1_tokens", 0) if compression_event else 0,
            "tool_calls_in_response": len(tool_calls),
        }
        state.timing_log.append(timing_entry)

        prompt_entry = {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "turn": state.turn_count,
            "step": state.step_count,
            "compressed_this_step": compression_event is not None,
            "input_tokens_est": prompt_tokens_est,
            "messages": prompt_dicts,
            "message_tokens": message_token_rows(prompt_dicts),
            "response": {
                "content": assistant_message.content,
                "tool_calls": tool_calls,
                "usage": usage,
                "cost": assistant_message.cost,
            },
        }
        state.prompt_entries.append(prompt_entry)
        state.turn_traces.append(
            {
                "sample_id": self.sample_id,
                "task_id": self.task_id,
                "turn": state.turn_count,
                "step": state.step_count,
                "prompt_tokens": prompt_tokens_est,
                "output_tokens": output_tokens,
                "compressed_this_step": compression_event is not None,
                "compression_index": compression_event.get("compression_index") if compression_event else None,
                "tool_calls": tool_calls,
                "assistant_content_preview": (assistant_message.content or "")[:1000],
            }
        )
        self._persist_logs(state, completed=False)
        self._save_checkpoint(state, completed=False)
        state.last_was_compression = False
        return assistant_message

    def _save_checkpoint(self, state: TelecomSPAgentState, completed: bool) -> None:
        payload = {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "completed": completed,
            "mode": self.mode_label,
            "turn_count": state.turn_count,
            "step_count": state.step_count,
            "compression_count": state.compression_count,
            "tool_call_count": state.tool_call_count,
            "reference_context_tokens": state.reference_context_tokens,
            "messages": self._messages_to_dicts(state.system_messages + state.messages),
            "previous_summary": state.previous_summary,
            "saved_at": time.time(),
        }
        path = self.checkpoint_dir / f"{self.sample_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def _persist_logs(self, state: TelecomSPAgentState, completed: bool) -> None:
        prompt_path = self.prompt_log_dir / f"{self.sample_id}.jsonl"
        with open(prompt_path, "w", encoding="utf-8") as handle:
            for entry in state.prompt_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        with open(self.abc_dir / f"{self.sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(state.abc_snapshots, handle, indent=2, ensure_ascii=False)

        with open(self.timing_dir / f"{self.sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(state.timing_log, handle, indent=2, ensure_ascii=False)

        trace = {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "mode": self.mode_label,
            "completed": completed,
            "total_turns": state.turn_count,
            "total_steps": state.step_count,
            "tool_calls": state.tool_call_count,
            "compressions": state.compression_count,
            "reference_context_tokens": state.reference_context_tokens,
            "turns": state.turn_traces,
        }
        with open(self.trace_dir / f"{self.sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(trace, handle, indent=2, ensure_ascii=False)


class TelecomSPBaselineAgent(TelecomSPAgent):
    def __init__(self, **kwargs):
        super().__init__(compress_enabled=False, **kwargs)


class TelecomSPCompressedAgent(TelecomSPAgent):
    def __init__(self, **kwargs):
        super().__init__(compress_enabled=True, **kwargs)


def _agent_kwargs(tools, domain_policy, kwargs: dict, mode_label: str) -> dict:
    llm_args = dict(kwargs.get("llm_args") or {})
    model_key = llm_args.pop("model_key", CFG.DEFAULT_MODEL)
    run_root = llm_args.pop("run_root", str(CFG.RESULTS_DIR / "adhoc"))
    initial_context_mode = llm_args.pop("initial_context_mode", CFG.INITIAL_CONTEXT_MODE)
    target_initial_tokens = int(llm_args.pop("target_initial_tokens", CFG.TARGET_INITIAL_TOKENS))
    context_window = int(llm_args.pop("context_window", CFG.CONTEXT_WINDOW))
    reserve_tokens = int(llm_args.pop("reserve_tokens", CFG.RESERVE_TOKENS))
    keep_recent_tokens = int(llm_args.pop("keep_recent_tokens", CFG.KEEP_RECENT_TOKENS))
    summary_max_tokens = int(llm_args.pop("summary_max_tokens", CFG.SUMMARY_MAX_TOKENS))
    include_task_ticket = bool(llm_args.pop("include_task_ticket", CFG.INCLUDE_TASK_TICKET))
    stepwise_tech_support = bool(llm_args.pop("stepwise_tech_support", CFG.STEPWISE_TECH_SUPPORT))
    task = kwargs.get("task")
    task_id = task.id if task is not None else "unknown_task"
    task_ticket = task.ticket if task is not None else None
    return {
        "tools": tools,
        "domain_policy": domain_policy,
        "llm": kwargs.get("llm"),
        "llm_args": llm_args,
        "model_key": model_key,
        "task_id": task_id,
        "run_root": run_root,
        "mode_label": mode_label,
        "initial_context_mode": initial_context_mode,
        "target_initial_tokens": target_initial_tokens,
        "context_window": context_window,
        "reserve_tokens": reserve_tokens,
        "keep_recent_tokens": keep_recent_tokens,
        "summary_max_tokens": summary_max_tokens,
        "include_task_ticket": include_task_ticket,
        "task_ticket": task_ticket,
        "stepwise_tech_support": stepwise_tech_support,
    }


def create_baseline_agent(tools, domain_policy, **kwargs):
    return TelecomSPBaselineAgent(**_agent_kwargs(tools, domain_policy, kwargs, "baseline"))


def create_compressed_agent(tools, domain_policy, **kwargs):
    return TelecomSPCompressedAgent(**_agent_kwargs(tools, domain_policy, kwargs, "compressed"))


def register_agents() -> None:
    from tau2.registry import registry

    registry.register_agent_factory(create_baseline_agent, "telecom_sp_baseline_agent")
    registry.register_agent_factory(create_compressed_agent, "telecom_sp_compressed_agent")
